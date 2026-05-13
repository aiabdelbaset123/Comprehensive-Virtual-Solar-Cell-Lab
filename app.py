import streamlit as st
import os, time, random, json, tempfile, textwrap, re, io, base64, copy, html, threading, traceback
from datetime import datetime
from typing import List, Dict, Any
import requests
import numpy as np
from scipy.constants import k, e
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from rdkit import Chem

try:
    from pymatgen.ext.matproj import MPRester
except ModuleNotFoundError:
    MPRester = None

try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ModuleNotFoundError:
    autogen = None
    AssistantAgent = UserProxyAgent = GroupChat = GroupChatManager = None

try:
    from deep_translator import GoogleTranslator
except ModuleNotFoundError:
    GoogleTranslator = None

# ===================== إعدادات الصفحة والترجمة =====================
st.set_page_config(page_title="المختبر الشامل", layout="wide")

def translate_ui(key, lang="ar"):
    """ترجمة النصوص الثابتة للواجهة"""
    translations = {
        "ar": {
            "title": "🧪 المختبر الافتراضي الشامل للخلايا الشمسية",
            "subtitle": "**جميع الميزات – هيكل بحثي، ML، ترجمة، قواعد بيانات متعددة**",
            "sidebar_header": "👤 معلومات الباحث",
            "researcher_name": "الاسم الكامل",
            "specialization": "التخصص",
            "ip_rights": "حقوق الملكية الفكرية",
            "task_theme": "محور البحث",
            "strict_mode": "🛡️ الوضع الصارم",
            "start_mission": "🚀 ابدأ المهمة",
            "agents_management": "👥 إدارة الوكلاء",
            "add_agent": "➕ إضافة وكيل",
            "agent_name": "اسم الوكيل",
            "system_message": "تعليمات النظام",
            "tools": "الأدوات",
            "save": "💾 حفظ",
            "delete": "🗑️ حذف",
            "sessions": "📚 الجلسات السابقة",
            "mission_title": "📋 تنفيذ المهمة",
            "chat": "💬 الحوار بين الوكلاء",
            "downloads": "📥 الملفات والنتائج",
            "dft_json": "⬇️ بيانات DFT (JSON)",
            "scaps_file": "⬇️ ملف SCAPS-1D",
            "pdf_report": "📄 تحميل تقرير PDF",
            "poster": "🖼️ ملصق المؤتمر",
            "param_sweep": "⬇️ رسم المسح البارامتري",
        },
        "en": {
            "title": "🧪 Comprehensive Virtual Solar Cell Lab",
            "subtitle": "**All Features – Research Hierarchy, ML, Translation, Multi-DB**",
            "sidebar_header": "👤 Researcher Info",
            "researcher_name": "Full Name",
            "specialization": "Specialization",
            "ip_rights": "Intellectual Property Rights",
            "task_theme": "Research Theme",
            "strict_mode": "🛡️ Strict Mode",
            "start_mission": "🚀 Start Mission",
            "agents_management": "👥 Agent Management",
            "add_agent": "➕ Add Agent",
            "agent_name": "Agent Name",
            "system_message": "System Message",
            "tools": "Tools",
            "save": "💾 Save",
            "delete": "🗑️ Delete",
            "sessions": "📚 Previous Sessions",
            "mission_title": "📋 Running Mission",
            "chat": "💬 Agent Conversation",
            "downloads": "📥 Files and Results",
            "dft_json": "⬇️ DFT Data (JSON)",
            "scaps_file": "⬇️ SCAPS-1D File",
            "pdf_report": "📄 Download PDF Report",
            "poster": "🖼️ Conference Poster",
            "param_sweep": "⬇️ Parametric Sweep Plot",
        }
    }
    return translations[lang].get(key, key)

def translate_text(text, source='auto', target='ar'):
    if GoogleTranslator is None:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except:
        return text

# اختيار اللغة
lang = st.sidebar.selectbox("🌐 اللغة / Language", ["العربية", "English"], index=0)
lang_code = "ar" if lang == "العربية" else "en"
T = lambda key: translate_ui(key, lang_code)

st.title(T("title"))
st.markdown(T("subtitle"))

# ===================== الأسرار =====================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    MP_API_KEY = st.secrets["MP_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MP_API_KEY = os.getenv("MP_API_KEY")

if not GEMINI_API_KEY or not MP_API_KEY:
    st.error("API keys are not configured. Add GEMINI_API_KEY and MP_API_KEY to Streamlit secrets or environment variables.")
    st.stop()

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# ===================== إعداد LLM =====================
config_list = [{"model": "gemini-1.5-flash", "api_type": "google", "api_key": GEMINI_API_KEY, "temperature": 0.3}]
llm_config = {"config_list": config_list, "timeout": 120, "cache_seed": None}

# ===================== حالة الجلسة =====================
if 'agents_config' not in st.session_state:
    st.session_state.agents_config = [
        {"name": "الباحث_الرئيسي", "system_message_ar": "أنت الباحث الرئيسي. تدير الفريق وتربط بين الأدوار. تطلب من كل وكيل القيام بمهمته، وتلخص النتائج. بعد كل وكيل، أطلب من مدقق الحقائق التحقق. في النهاية أصدر تقريراً نهائياً. تحدث بالعربية.",
         "system_message_en": "You are the Principal Investigator. Coordinate the team, request analyses from each expert, summarize after each step, and produce the final report. Ask the Fact-Checker to verify after each major result. Speak in English.", "tools": []},
        {"name": "عالم_المواد_الحاسوبي", "system_message_ar": "ابحث في Materials Project, OQMD, AFLOW, NOMAD عن مواد خالية من الرصاص بفجوة 1.2-1.6 eV. احسب عامل التسامح. قارن بين 3 مواد من مصادر مختلفة. حلل أفضلها DFT. لا تذكر مادة قبل استخدام الأداة. تحدث بالعربية.",
         "system_message_en": "Search Materials Project, OQMD, AFLOW, NOMAD for lead-free perovskites with bandgap 1.2-1.6 eV. Compute tolerance factor. Compare top 3 materials across sources. Run DFT on the best. Do not mention any material before using a tool. Speak in English.", "tools": ["search_materials_project", "search_oqmd", "search_aflow", "search_nomad", "calculate_tolerance_factor", "dft_analysis"]},
        {"name": "الكيميائي_النظري", "system_message_ar": "حلل الاستقرار الكيميائي والسمية بناءً على طاقة التشكل والطاقة فوق الهيكل. ناقش التأكسد. اقترح تحسينات. تحدث بالعربية.",
         "system_message_en": "Analyze chemical stability and toxicity from formation energy and energy above hull. Discuss oxidation risks. Suggest doping or alloying improvements. Speak in English.", "tools": []},
        {"name": "مهندس_الأجهزة", "system_message_ar": "استخدم scaps_simulation لمحاكاة الخلية. قم بمسح بارامتري للسمك (100-1000 nm). قارن بين درجات حرارة التشغيل (25,45,65,85 مئوية). تحدث بالعربية.",
         "system_message_en": "Use scaps_simulation to model the cell. Perform parametric sweep of thickness (100-1000 nm). Compare different operating temperatures. Speak in English.", "tools": ["scaps_simulation"]},
        {"name": "مهندس_التصنيع", "system_message_ar": "اقترح طرق تصنيع (Spin-coating, evaporation) وتقدير التكلفة. ناقش تحديات التوسع. تحدث بالعربية.",
         "system_message_en": "Propose fabrication methods (spin-coating, evaporation) and cost estimate. Discuss scalability challenges. Speak in English.", "tools": []},
        {"name": "الناقد_العلمي", "system_message_ar": "ابحث في Semantic Scholar عن أحدث أوراق 2024-2025. انتقد المنهجية والفجوات. تحدث بالعربية.",
         "system_message_en": "Search Semantic Scholar for latest 2024-2025 papers. Critique methodology and gaps. Speak in English.", "tools": ["search_semantic_scholar"]},
        {"name": "خبير_الملكية_الفكرية", "system_message_ar": "ابحث في براءات الاختراع (Google Patents) عن المادة المقترحة. قيم مخاطر التعارض مع حقوق الباحث. تحدث بالعربية.",
         "system_message_en": "Search patents (Google Patents) for the proposed material. Assess IP conflict risks. Speak in English.", "tools": ["search_google_patents"]},
        {"name": "مدقق_الحقائق", "system_message_ar": "بعد كل وكيل، تحقق من الأرقام بإعادة استدعاء الأداة المناسبة. إذا وجدت خطأ، صححه. تحدث بالعربية.",
         "system_message_en": "After each expert, verify numbers by re-running the relevant tool. Correct any errors. Speak in English.", "tools": ["search_materials_project", "dft_analysis", "scaps_simulation", "search_semantic_scholar"]},
        {"name": "عالم_تعلم_الآلة", "system_message_ar": "استخدم أدوات ML للتنبؤ بالكفاءة، واقتراح تركيبات جديدة، والكشف عن القيم الشاذة. تحدث بالعربية.",
         "system_message_en": "Use ML tools to predict efficiency, suggest novel compositions, and detect anomalies. Speak in English.", "tools": ["ml_predict_efficiency", "ml_generate_candidates", "ml_detect_anomalies"]}
    ]

if 'sessions' not in st.session_state:
    st.session_state.sessions = []

if 'strict_mode' not in st.session_state:
    st.session_state.strict_mode = False

# ===================== أدوات قواعد البيانات =====================
def search_materials_project(bandgap_min=1.2, bandgap_max=1.8, elements=None, n=5):
    if not MP_API_KEY: return "❌ MP key missing."
    try:
        url = "https://api.materialsproject.org/materials/summary/"
        filters = {"band_gap": [bandgap_min, bandgap_max], "nelements": [2,4], "material_ids":[]}
        if elements: filters["elements"] = elements
        payload = {"criteria": filters, "properties": ["material_id","formula_pretty","band_gap","formation_energy_per_atom","energy_above_hull","nsites","volume"]}
        resp = requests.post(url, json=payload, headers={"X-API-KEY": MP_API_KEY}, params={"_skip":0,"_limit":n,"_all_fields":False})
        resp.raise_for_status()
        data = resp.json().get("data",[])
        if not data: return "⚠️ No materials found."
        lines = [f"📊 Materials Project ({len(data)}):", ""]
        for i,item in enumerate(data,1):
            lines.append(f"**{i}. {item['formula_pretty']}** (id: {item['material_id']})")
            lines.append(f"  - Band gap: {item.get('band_gap')} eV")
            lines.append(f"  - Formation energy: {item.get('formation_energy_per_atom')} eV/atom")
            lines.append(f"  - Energy above hull: {item.get('energy_above_hull')} eV/atom")
            lines.append("")
        return "\n".join(lines)
    except Exception as e: return f"❌ Error: {e}"

def search_oqmd(bandgap_min=1.2, bandgap_max=1.8, n=5):
    try:
        # simplified, real API may need different format
        url = "https://oqmd.org/oqmdapi/formationenergy"
        params = {"band_gap_min": bandgap_min, "band_gap_max": bandgap_max, "limit": n}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json().get("data",[])
        if not data: return "⚠️ OQMD: no results."
        lines = [f"📊 OQMD ({len(data)})", ""]
        for item in data[:n]:
            lines.append(f"- {item.get('name','?')}: band gap {item.get('band_gap','?')} eV")
        return "\n".join(lines)
    except Exception as e: return f"❌ OQMD error: {e}"

def search_aflow(bandgap_min=1.2, bandgap_max=1.8, n=5):
    try:
        url = "http://aflowlib.org/API/aflux/"
        params = {"band_gap_min": bandgap_min, "band_gap_max": bandgap_max, "limit": n}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        lines = [f"📊 AFLOW ({len(data)})", ""]
        for entry in data[:n]:
            lines.append(f"- {entry.get('compound','?')}: band gap {entry.get('Egap','?')} eV")
        return "\n".join(lines)
    except Exception as e: return f"❌ AFLOW error: {e}"

def search_nomad(material_name, n=5):
    try:
        url = "https://nomad-lab.eu/prod/v1/api/v1/entries/query"
        resp = requests.post(url, json={"query": {"results.material.elements": material_name}}, params={"per_page": n})
        resp.raise_for_status()
        data = resp.json().get("data",[])
        lines = [f"📊 NOMAD ({len(data)})", ""]
        for item in data[:n]:
            lines.append(f"- {item.get('material','?')}: DFT data available")
        return "\n".join(lines)
    except Exception as e: return f"❌ NOMAD error: {e}"

def search_semantic_scholar(query:str, limit=5):
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query":query,"limit":limit,"fields":"title,authors,year,externalIds,abstract"}
        resp = requests.get(url, params=params, headers={"Accept":"application/json"})
        resp.raise_for_status()
        data = resp.json().get("data",[])
        if not data: return f"⚠️ No papers for: {query}"
        lines = [f"🔬 Semantic Scholar ({len(data)})", ""]
        for i,paper in enumerate(data,1):
            title = paper.get("title","No title")
            year = paper.get("year","?")
            authors = [a.get("name","") for a in paper.get("authors",[])]
            authors_str = ", ".join(authors[:3]) + (" et al." if len(authors)>3 else "")
            abstract = paper.get("abstract","")
            abstract_short = (abstract[:300]+"...") if abstract and len(abstract)>300 else abstract or "No abstract"
            lines.append(f"**{i}. {title}**")
            lines.append(f"  - Authors: {authors_str} ({year})")
            lines.append(f"  - Abstract: {abstract_short}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e: return f"❌ Error: {e}"

def search_google_patents(query, limit=3):
    # simulated
    return f"🔍 Patents about '{query}': (simulation) 3 patents found. Please verify manually."

def calculate_tolerance_factor(rA, rB, rX):
    import math
    t = (rA + rX) / (math.sqrt(2) * (rB + rX))
    t = round(t,3)
    if 0.8 <= t <= 1.0: stability = "stable"
    elif 0.7 <= t < 0.8: stability = "semi-stable"
    elif 1.0 < t <= 1.1: stability = "semi-stable (hexagonal)"
    else: stability = "unstable"
    return f"🧮 Tolerance factor t = {t} → {stability}"

def dft_analysis(material_id):
    if MPRester is None:
        return {
            "report": "pymatgen is not installed. Install dependencies with `python -m pip install -r requirements.txt` to enable DFT analysis.",
            "json_data": None,
        }
    if not MP_API_KEY: return {"report": "❌ MP key missing.", "json_data": None}
    try:
        with MPRester(MP_API_KEY) as mpr:
            data = mpr.materials.search(material_ids=[material_id],
                fields=["material_id","formula_pretty","band_gap","formation_energy_per_atom","energy_above_hull","dielectric","effective_mass"])
            if not data: return {"report": f"❌ Material {material_id} not found.", "json_data": None}
            mat = data[0]
            bg = getattr(mat,'band_gap',None)
            e_above_hull = getattr(mat,'energy_above_hull',None)
            form_energy = getattr(mat,'formation_energy_per_atom',None)
            eff_mass = getattr(mat,'effective_mass',None)
            dielectric = getattr(mat,'dielectric',None)
            stability = "stable" if e_above_hull and e_above_hull<0.05 else "semi-stable" if e_above_hull and e_above_hull<0.1 else "unstable"
            json_data = {"material_id":material_id,"formula":mat.formula_pretty,"band_gap_GGA":bg,
                         "energy_above_hull":e_above_hull,"formation_energy":form_energy,
                         "effective_mass":eff_mass,"dielectric":dielectric,"stability":stability}
            report = f"""
🔬 **DFT Analysis for {mat.formula_pretty} ({material_id}):**
- Band gap (GGA): {bg:.2f} eV
- Energy above hull: {e_above_hull:.3f} eV/atom → {stability}
- Formation energy: {form_energy:.3f} eV/atom
- Dielectric: {dielectric}
"""
            if eff_mass: report += f"- Effective mass: e={eff_mass.get('e')} mₑ, h={eff_mass.get('h')} mₑ"
            return {"report": report, "json_data": json_data}
    except Exception as e: return {"report": f"❌ DFT failed: {e}", "json_data": None}

def scaps_simulation(material_formula, bandgap, electron_affinity=3.9, permittivity=30.0,
                     thickness=0.6, N_A=3e15, N_D=1e15, mu_n=10.0, mu_p=10.0,
                     tau_n=1e-6, tau_p=1e-6, R_shunt=1000.0, R_series=1.0,
                     temperature=300.0, illumination="AM1.5G") -> dict:
    try:
        kT_q = k * temperature / e
        Eg = bandgap
        sq_data = {1.1:42,1.2:38,1.3:35,1.4:32,1.5:30,1.6:28,1.7:26,1.8:24,1.9:22}
        Jsc_est = sq_data.get(round(Eg,1), 25)
        J0_rad = 1.5e-14 * np.exp(-Eg / kT_q)
        Voc_rad = kT_q * np.log(Jsc_est*1e-3 / J0_rad + 1)
        Voc_real = 0.85 * Voc_rad
        n = 1.5
        voc_norm = Voc_real / (n * kT_q)
        FF = (voc_norm - np.log(voc_norm + 0.72)) / (voc_norm + 1)
        FF = max(0.7, min(0.85, FF))
        PCE = Jsc_est * Voc_real * FF / 10

        scaps_content = f"""
[SCAPS version 3.3.10]
[Device]
left contact = front
right contact = back
[Layers]
layer 1 = {material_formula}
thickness 1 = {thickness} um
bandgap 1 = {bandgap} eV
affinity 1 = {electron_affinity} eV
permittivity 1 = {permittivity}
Nc 1 = 2.2e18 cm-3
Nv 1 = 1.8e19 cm-3
mu_n 1 = {mu_n} cm2/Vs
mu_p 1 = {mu_p} cm2/Vs
tau_n 1 = {tau_n} s
tau_p 1 = {tau_p} s
N_A 1 = {N_A} cm-3
N_D 1 = {N_D} cm-3
[Defects]
defect 1 = single donor
Nt 1 = 1e15 cm-3
Et 1 = 0.6 eV (from CB)
sigma_n 1 = 1e-14 cm2
sigma_p 1 = 1e-15 cm2
[Illumination]
spectrum = {illumination}
[Contacts]
left workfunction = 4.2 eV
right workfunction = 5.0 eV
R_shunt = {R_shunt} ohm cm2
R_series = {R_series} ohm cm2
"""
        report = f"""
🖥️ **SCAPS Simulation:**
⚡ Band gap: {Eg:.2f} eV
☀️ Jsc: {Jsc_est:.1f} mA/cm²
🔋 Voc: {Voc_real:.3f} V
📊 FF: {FF:.3f}
🌟 Efficiency: {PCE:.1f}%
"""
        return {"report": report, "scaps_file": textwrap.dedent(scaps_content), "pce": PCE}
    except Exception as e:
        return {"report": f"❌ SCAPS failed: {e}", "scaps_file": None, "pce": 0}

# ===================== أدوات تعلم الآلة =====================
@st.cache_resource
def load_efficiency_model():
    """نموذج انحدار بسيط تم تدريبه مسبقاً (بيانات وهمية لكن يمكن استبدالها)"""
    X = np.array([[1.2,0.2,0.02,30],[1.5,0.3,0.05,25],[1.8,0.25,0.1,20],[1.34,0.18,0.03,28],
                  [1.6,0.22,0.08,22],[1.3,0.19,0.04,27]])
    y = np.array([22.5,20.1,16.3,24.0,19.0,23.5])
    model = RandomForestRegressor(n_estimators=50)
    model.fit(X, y)
    return model

def ml_predict_efficiency(bandgap, eff_mass_e, energy_above_hull, dielectric):
    try:
        model = load_efficiency_model()
        pred = model.predict([[bandgap, eff_mass_e, energy_above_hull, dielectric]])[0]
        return f"🤖 ML predicted efficiency: {pred:.1f}%"
    except Exception as e:
        return f"❌ ML prediction error: {e}"

def ml_generate_candidates(n=3):
    """توليد تركيبات عضوية-لاعضوية عشوائية (بسيطة)"""
    try:
        templates = ["C[NH2+]C.[I-]", "C[NH2+].[Br-]", "CC[NH2+]C.[I-]", "C1=CC=C[NH2+]1.[I-]"]
        mols = [Chem.MolFromSmiles(s) for s in random.sample(templates, min(n, len(templates)))]
        smiles = [Chem.MolToSmiles(m) for m in mols]
        return f"🧪 ML suggested candidates (SMILES): {', '.join(smiles)}"
    except Exception as e:
        return f"❌ ML generation error: {e}"

def ml_detect_anomalies(data_points: str):
    """كشف القيم الشاذة في سلسلة أرقام"""
    try:
        arr = np.fromstring(data_points, sep=',')
        if len(arr) < 3: return "Need at least 3 values."
        clf = IsolationForest(contamination=0.2)
        pred = clf.fit_predict(arr.reshape(-1,1))
        anomalies = arr[pred == -1]
        return f"🔎 ML anomaly detection: anomalous values = {list(anomalies)}"
    except Exception as e:
        return f"❌ ML anomaly error: {e}"

# سجل الأدوات
AVAILABLE_TOOLS = {
    "search_materials_project": search_materials_project,
    "search_oqmd": search_oqmd,
    "search_aflow": search_aflow,
    "search_nomad": search_nomad,
    "search_semantic_scholar": search_semantic_scholar,
    "search_google_patents": search_google_patents,
    "calculate_tolerance_factor": calculate_tolerance_factor,
    "dft_analysis": dft_analysis,
    "scaps_simulation": scaps_simulation,
    "ml_predict_efficiency": ml_predict_efficiency,
    "ml_generate_candidates": ml_generate_candidates,
    "ml_detect_anomalies": ml_detect_anomalies
}

# ===================== بناء الوكلاء الديناميكي =====================
def build_agents(strict=False, lang_code="ar"):
    agents = []
    for agent_cfg in st.session_state.agents_config:
        sys_msg_key = "system_message_ar" if lang_code == "ar" else "system_message_en"
        sys_msg = agent_cfg.get(sys_msg_key, agent_cfg.get("system_message_ar", ""))
        if strict:
            sys_msg += "\n\n[STRICT MODE: You MUST use tools before any conclusion. Cite sources.]"
        agent = AssistantAgent(
            name=agent_cfg["name"],
            system_message=sys_msg,
            llm_config=llm_config,
        )
        tools = {}
        for tool_name in agent_cfg["tools"]:
            if tool_name in AVAILABLE_TOOLS:
                tools[tool_name] = AVAILABLE_TOOLS[tool_name]
        if tools:
            agent.register_function(function_map=tools)
        agents.append(agent)
    return agents

# أيقونات
AGENT_AVATARS = {
    "الباحث_الرئيسي": "🧑‍💼",
    "عالم_المواد_الحاسوبي": "🧑‍🔬",
    "الكيميائي_النظري": "⚗️",
    "مهندس_الأجهزة": "⚡",
    "مهندس_التصنيع": "🏭",
    "الناقد_العلمي": "🔬",
    "خبير_الملكية_الفكرية": "📜",
    "مدقق_الحقائق": "✅",
    "عالم_تعلم_الآلة": "🤖",
}

def _agent_display_name(name):
    return str(name or "unknown").replace("_", " ")

def _message_content(message):
    content = message.get("content", "")
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content or "").strip()

def render_live_discussion(messages, started_at, is_running=True, error=None):
    elapsed = max(0, int(time.time() - started_at))
    active_name = messages[-1].get("name", messages[-1].get("role", "بانتظار الرد")) if messages else "بانتظار أول رد"
    unique_speakers = {msg.get("name", msg.get("role", "unknown")) for msg in messages}
    state_label = "قيد التنفيذ" if is_running else "اكتملت المهمة"

    st.markdown(
        """
        <style>
        .mission-hero {
            border: 1px solid rgba(120, 128, 145, 0.28);
            border-radius: 8px;
            padding: 18px 20px;
            margin: 8px 0 16px 0;
            background: linear-gradient(135deg, rgba(18, 28, 38, 0.96), rgba(31, 45, 54, 0.96));
            color: #f8fafc;
        }
        .mission-hero h3 { margin: 0 0 6px 0; font-size: 1.15rem; letter-spacing: 0; }
        .mission-hero p { margin: 0; color: #cbd5e1; }
        .agent-strip {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 4px 0 14px 0;
        }
        .agent-chip {
            border: 1px solid rgba(120, 128, 145, 0.25);
            border-radius: 8px;
            padding: 7px 10px;
            background: #ffffff;
            color: #172033;
            font-size: 0.88rem;
        }
        .agent-chip.active {
            border-color: #0f766e;
            box-shadow: inset 0 0 0 1px rgba(15, 118, 110, 0.25);
        }
        .live-dot {
            display: inline-block;
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: #22c55e;
            margin-inline-end: 8px;
            vertical-align: middle;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="mission-hero">
            <h3><span class="live-dot"></span>لوحة تنفيذ المهمة المباشرة</h3>
            <p>الحوار بين الوكلاء يظهر هنا أثناء التفكير، استخدام الأدوات، والتحقق من النتائج.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("الحالة", state_label)
    col2.metric("الزمن", f"{elapsed} ث")
    col3.metric("عدد الرسائل", len(messages))
    col4.metric("آخر متحدث", _agent_display_name(active_name))

    chips = []
    for agent_name, avatar in AGENT_AVATARS.items():
        active_class = " active" if agent_name in unique_speakers else ""
        chips.append(
            f'<span class="agent-chip{active_class}">{html.escape(avatar)} {html.escape(_agent_display_name(agent_name))}</span>'
        )
    st.markdown(f'<div class="agent-strip">{"".join(chips)}</div>', unsafe_allow_html=True)

    if error:
        st.error("توقف التنفيذ بسبب خطأ أثناء نقاش الوكلاء.")

    tab_chat, tab_summary = st.tabs(["الحوار المباشر", "ملخص التنفيذ"])
    with tab_chat:
        if not messages:
            st.info("تم إطلاق المهمة. ستظهر أول رسالة هنا فور صدورها من الوكلاء.")
        for index, msg in enumerate(messages, 1):
            name = msg.get("name", msg.get("role", "unknown"))
            avatar = AGENT_AVATARS.get(name, "🤖")
            with st.chat_message(name, avatar=avatar):
                st.caption(f"دور {index} · {_agent_display_name(name)}")
                content = _message_content(msg)
                st.markdown(content if content else "_لا توجد رسالة نصية في هذا الدور._")
                if msg.get("function_call"):
                    st.code(json.dumps(msg["function_call"], ensure_ascii=False, indent=2), language="json")

    with tab_summary:
        if not messages:
            st.write("لم تبدأ المخرجات بعد.")
        else:
            for name in sorted(unique_speakers):
                agent_messages = [m for m in messages if m.get("name", m.get("role", "unknown")) == name]
                latest = _message_content(agent_messages[-1]) if agent_messages else ""
                preview = re.sub(r"\s+", " ", latest)[:260]
                st.markdown(f"**{_agent_display_name(name)}** · {len(agent_messages)} رسالة")
                st.caption(preview if preview else "لا توجد خلاصة نصية بعد.")

# ===================== واجهة المستخدم =====================
with st.sidebar:
    st.header(T("sidebar_header"))
    researcher_name = st.text_input(T("researcher_name"), "د. أحمد الباحث")
    specialization = st.text_input(T("specialization"), "فيزياء المواد المكثفة")
    ip_rights = st.text_area(T("ip_rights"), "براءة رقم 12345\nحقوق نشر 2024")

    st.header("⚙️ مهمة البحث")
    task_themes = {
        "ar": ["بيروفسكايت هاليد مزدوج خالٍ من الرصاص", "تحسين استقرار بيروفسكايت القصدير", "خلية 1.34 eV", "مقارنة ETL"],
        "en": ["Lead-free double halide perovskite", "Tin-based stability improvement", "1.34 eV cell", "ETL comparison"]
    }
    task_theme = st.selectbox(T("task_theme"), task_themes[lang_code])

    st.session_state.strict_mode = st.checkbox(T("strict_mode"), value=False)

    if st.button(T("start_mission"), type="primary"):
        st.session_state.start_mission = True
    else:
        st.session_state.start_mission = False

    st.markdown("---")
    st.header(T("agents_management"))
    with st.expander(T("add_agent")):
        new_name = st.text_input(T("agent_name"), key="new_name")
        new_sys_ar = st.text_area("تعليمات (عربي)", key="new_sys_ar")
        new_sys_en = st.text_area("System message (EN)", key="new_sys_en")
        new_tools = st.multiselect(T("tools"), list(AVAILABLE_TOOLS.keys()), key="new_tools")
        if st.button("أضف / Add"):
            if new_name and (new_sys_ar or new_sys_en):
                st.session_state.agents_config.append({
                    "name": new_name,
                    "system_message_ar": new_sys_ar,
                    "system_message_en": new_sys_en,
                    "tools": new_tools
                })
                st.success(f"تمت إضافة {new_name}")
                st.experimental_rerun()
    for idx, agent in enumerate(st.session_state.agents_config):
        with st.expander(f"{AGENT_AVATARS.get(agent['name'], '🤖')} {agent['name']}"):
            col1, col2 = st.columns([3,1])
            with col1:
                edited_name = st.text_input("Name", agent['name'], key=f"name_{idx}")
                edited_sys_ar = st.text_area("Arabic msg", agent.get('system_message_ar',''), key=f"sys_ar_{idx}")
                edited_sys_en = st.text_area("English msg", agent.get('system_message_en',''), key=f"sys_en_{idx}")
                edited_tools = st.multiselect("Tools", list(AVAILABLE_TOOLS.keys()), default=agent['tools'], key=f"tools_{idx}")
            with col2:
                if st.button(T("delete"), key=f"del_{idx}"):
                    st.session_state.agents_config.pop(idx)
                    st.experimental_rerun()
                if st.button(T("save"), key=f"save_{idx}"):
                    st.session_state.agents_config[idx] = {
                        "name": edited_name,
                        "system_message_ar": edited_sys_ar,
                        "system_message_en": edited_sys_en,
                        "tools": edited_tools
                    }
                    st.success("تم الحفظ / Saved!")
    st.markdown("---")
    st.header(T("sessions"))
    for i, sess in enumerate(st.session_state.sessions):
        st.write(f"{i+1}. {sess['time'][:10]} - {sess.get('material','?')} ({sess.get('pce',0):.1f}%)")

# ===================== تنفيذ المهمة =====================
if st.session_state.get("start_mission", False):
    st.header(T("mission_title"))
    if autogen is None:
        st.error("pyautogen is not installed. Install dependencies with `python -m pip install -r requirements.txt` to run the agent mission.")
        st.stop()
    agents = build_agents(strict=st.session_state.strict_mode, lang_code=lang_code)
    user_proxy = UserProxyAgent(
        name="human",
        human_input_mode="NEVER",
        llm_config=False,
        code_execution_config={"use_docker": False},
    )
    all_agents = [user_proxy] + agents

    # تخصيص ترتيب المتحدثين
    def custom_speaker_selection(last_speaker, groupchat):
        # الباحث الرئيسي يتحدث أولاً، ثم حسب الترتيب المنطقي
        order = ["الباحث_الرئيسي", "عالم_المواد_الحاسوبي", "الكيميائي_النظري", "مهندس_الأجهزة",
                 "مهندس_التصنيع", "عالم_تعلم_الآلة", "الناقد_العلمي", "خبير_الملكية_الفكرية", "مدقق_الحقائق"]
        agents_by_name = {agent.name: agent for agent in groupchat.agents}
        # العودة إلى الباحث الرئيسي بعد كل دورة
        if last_speaker is None:
            return agents_by_name.get(order[0], "round_robin")
        idx = order.index(last_speaker.name) if last_speaker.name in order else -1
        next_idx = (idx + 1) % len(order)
        return agents_by_name.get(order[next_idx], "round_robin")

    groupchat = GroupChat(
        agents=all_agents,
        messages=[],
        max_round=40,
        speaker_selection_method=custom_speaker_selection,
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    # رسالة البداية (حسب اللغة)
    if lang_code == "ar":
        initial_message = f"""
🎯 مهمة: {task_theme}
المجموعة البحثية بقيادة الباحث الرئيسي.
سيتحدث كل خبير بدوره. يرجى الالتزام بالأدوات والتحقق.
معلومات الباحث: {researcher_name} - {specialization}
حقوق: {ip_rights}
        """
    else:
        initial_message = f"""
🎯 Mission: {task_theme}
Research group led by Principal Investigator. Each expert will speak in turn. Use tools and verify.
Researcher: {researcher_name} - {specialization}
IP rights: {ip_rights}
        """

    start = time.time()
    mission_state = {"done": False, "error": None, "traceback": None}

    def run_agent_mission():
        try:
            user_proxy.initiate_chat(manager, message=initial_message, silent=True)
        except Exception as exc:
            mission_state["error"] = str(exc)
            mission_state["traceback"] = traceback.format_exc()
        finally:
            mission_state["done"] = True

    live_panel = st.empty()
    mission_thread = threading.Thread(target=run_agent_mission, daemon=True)
    mission_thread.start()

    while mission_thread.is_alive():
        with live_panel.container():
            render_live_discussion(list(groupchat.messages), start, is_running=True)
        time.sleep(1.0)

    mission_thread.join()
    messages = list(groupchat.messages)
    duration = time.time()-start
    with live_panel.container():
        render_live_discussion(messages, start, is_running=False, error=mission_state["error"])

    if mission_state["error"]:
        st.code(mission_state["traceback"], language="python")
        st.stop()

    st.success(f"تم إنجاز المهمة في {duration:.0f} ثانية")

    # استخراج البيانات للملفات
    last_material_id = last_formula = last_bandgap = None
    for msg in reversed(messages):
        if msg['name'] == 'عالم_المواد_الحاسوبي':
            ids = re.findall(r'mp-\d+', msg['content'])
            if ids:
                last_material_id = ids[-1]
                formulas = re.findall(r'[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+', msg['content'])
                if formulas: last_formula = formulas[0]
                bg_match = re.search(r'[Bb]and.?gap.*?(\d+\.\d+)', msg['content'])
                if bg_match: last_bandgap = float(bg_match.group(1))
            break

    st.header(T("downloads"))
    if last_material_id and last_formula and last_bandgap:
        # DFT
        dft_res = dft_analysis(last_material_id)
        if dft_res['json_data']:
            st.download_button(T("dft_json"), json.dumps(dft_res['json_data'], indent=2),
                               f"dft_{last_material_id}.json")
        # SCAPS
        scaps_res = scaps_simulation(last_formula, last_bandgap)
        if scaps_res['scaps_file']:
            st.download_button(T("scaps_file"), scaps_res['scaps_file'], f"scaps_{last_formula}.scaps")
        # مسح بارامتري
        thicknesses = np.linspace(100,1000,10)
        pces = []
        for th in thicknesses:
            sim = scaps_simulation(last_formula, last_bandgap, thickness=th/1000)
            pces.append(sim['pce'])
        fig, ax = plt.subplots()
        ax.plot(thicknesses, pces)
        ax.set_xlabel("Thickness (nm)"); ax.set_ylabel("Efficiency (%)")
        buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
        st.pyplot(fig)
        st.download_button(T("param_sweep"), buf.getvalue(), "sweep.png")
        # PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, text=f"Report: {last_formula}", ln=True)
        pdf.output("/tmp/report.pdf")
        with open("/tmp/report.pdf","rb") as f:
            st.download_button(T("pdf_report"), f, "report.pdf")
        # Poster
        img = Image.new('RGB',(800,400),'white')
        d = ImageDraw.Draw(img)
        d.text((50,50), f"Material: {last_formula}", fill='black')
        d.text((50,100), f"Efficiency: {scaps_res['pce']:.1f}%", fill='black')
        buf = io.BytesIO(); img.save(buf, format='PNG'); buf.seek(0)
        st.download_button(T("poster"), buf.getvalue(), "poster.png")

        # حفظ الجلسة
        st.session_state.sessions.append({
            "time": datetime.now().isoformat(),
            "theme": task_theme,
            "material": last_formula,
            "pce": scaps_res['pce']
        })
