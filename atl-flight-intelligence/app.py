"""ATL Flight Intelligence — dashboard penerbangan minimalis mode gelap.

Perubahan hanya pada presentasi UI. Proses verifikasi bundle, pemilihan data,
inference LightGBM, ambang validasi, cutoff T-2h, dan SHAP tetap berasal
dari inference.py serta explain.py milik proyek penelitian.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from inference import predict, verify_bundle
from explain import local_waterfall


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "Model_4_3_2_Terpilih_Streamlit.joblib"
DATA_PATH = ROOT / "data" / "testing_flights_2024.csv"

MONTHS_ID = (
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
)


# Nama maskapai di bawah mengikuti IDENTITAS OPERATOR (op_unique_carrier),
# bukan selalu merek pemasaran/codeshare yang tertera pada tiket.
# Rujukan: BTS dan FAA ASQP (khususnya operator regional pada tahun 2024).
CARRIER_NAMES = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "NK": "Spirit Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "QX": "Horizon Air",
    "SY": "Sun Country Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "YX": "Republic Airways",
    "YV": "Mesa Airlines",
    "ZW": "Air Wisconsin Airlines",
    "PT": "Piedmont Airlines",
    "C5": "CommuteAir",
    "G7": "GoJet Airlines",
}

# Nama bandara untuk rute domestik AS, termasuk tujuan umum dari ATL.
# Data referensi tampilan, TIDAK digunakan oleh model atau feature engineering.
# Untuk kode yang belum ada, UI menggunakan kode asli tanpa menebak nama.
AIRPORT_NAMES = dict(
    line.split("|", 1)
    for line in """ABE|Lehigh Valley International Airport
ABI|Abilene Regional Airport
ABQ|Albuquerque International Sunport
ABR|Aberdeen Regional Airport
ABY|Southwest Georgia Regional Airport
ACK|Nantucket Memorial Airport
ACT|Waco Regional Airport
ACV|California Redwood Coast–Humboldt County Airport
ACY|Atlantic City International Airport
ADK|Adak Airport
ADQ|Kodiak Airport
AEX|Alexandria International Airport
AGS|Augusta Regional Airport
ALB|Albany International Airport
ALO|Waterloo Regional Airport
AMA|Rick Husband Amarillo International Airport
ANC|Ted Stevens Anchorage International Airport
APN|Alpena County Regional Airport
ART|Watertown International Airport
ASE|Aspen/Pitkin County Airport
ATL|Hartsfield–Jackson Atlanta International Airport
ATW|Appleton International Airport
AUS|Austin–Bergstrom International Airport
AVL|Asheville Regional Airport
AVP|Wilkes-Barre/Scranton International Airport
AZO|Kalamazoo/Battle Creek International Airport
BDL|Bradley International Airport
BET|Bethel Airport
BFL|Meadows Field Airport
BGM|Greater Binghamton Airport
BGR|Bangor International Airport
BHM|Birmingham–Shuttlesworth International Airport
BIL|Billings Logan International Airport
BIS|Bismarck Municipal Airport
BJI|Bemidji Regional Airport
BLI|Bellingham International Airport
BMI|Central Illinois Regional Airport
BNA|Nashville International Airport
BOI|Boise Airport
BOS|Boston Logan International Airport
BPT|Jack Brooks Regional Airport
BQK|Brunswick Golden Isles Airport
BRD|Brainerd Lakes Regional Airport
BRO|Brownsville/South Padre Island International Airport
BRW|Wiley Post–Will Rogers Memorial Airport
BTM|Bert Mooney Airport
BTR|Baton Rouge Metropolitan Airport
BTV|Burlington International Airport
BUF|Buffalo Niagara International Airport
BUR|Hollywood Burbank Airport
BWI|Baltimore/Washington International Thurgood Marshall Airport
BZN|Bozeman Yellowstone International Airport
CAE|Columbia Metropolitan Airport
CAK|Akron–Canton Airport
CDC|Cedar City Regional Airport
CDV|Merle K. (Mudhole) Smith Airport
CGI|Cape Girardeau Regional Airport
CHA|Chattanooga Metropolitan Airport
CHO|Charlottesville–Albemarle Airport
CHS|Charleston International Airport
CID|The Eastern Iowa Airport
CIU|Chippewa County International Airport
CKB|North Central West Virginia Airport
CLE|Cleveland Hopkins International Airport
CLL|Easterwood Airport
CLT|Charlotte Douglas International Airport
CMH|John Glenn Columbus International Airport
CMI|University of Illinois Willard Airport
CMX|Houghton County Memorial Airport
COD|Yellowstone Regional Airport
COS|Colorado Springs Airport
COU|Columbia Regional Airport
CPR|Casper–Natrona County International Airport
CRP|Corpus Christi International Airport
CRW|West Virginia International Yeager Airport
CSG|Columbus Airport (Georgia)
CVG|Cincinnati/Northern Kentucky International Airport
CWA|Central Wisconsin Airport
DAB|Daytona Beach International Airport
DAL|Dallas Love Field
DAY|James M. Cox Dayton International Airport
DBQ|Dubuque Regional Airport
DCA|Ronald Reagan Washington National Airport
DEN|Denver International Airport
DFW|Dallas Fort Worth International Airport
DHN|Dothan Regional Airport
DIK|Dickinson Theodore Roosevelt Regional Airport
DLG|Dillingham Airport
DLH|Duluth International Airport
DRO|Durango–La Plata County Airport
DSM|Des Moines International Airport
DTW|Detroit Metropolitan Wayne County Airport
DUJ|DuBois Regional Airport
DUT|Unalaska Airport
EAT|Pangborn Memorial Airport
EAU|Chippewa Valley Regional Airport
ECP|Northwest Florida Beaches International Airport
EGE|Eagle County Regional Airport
EKO|Elko Regional Airport
ELM|Elmira/Corning Regional Airport
ELP|El Paso International Airport
ERI|Erie International Airport
ESC|Delta County Airport
EUG|Eugene Airport
EVV|Evansville Regional Airport
EWN|Coastal Carolina Regional Airport
EWR|Newark Liberty International Airport
EYW|Key West International Airport
FAI|Fairbanks International Airport
FAR|Hector International Airport
FAT|Fresno Yosemite International Airport
FAY|Fayetteville Regional Airport
FCA|Glacier Park International Airport
FLG|Flagstaff Pulliam Airport
FLL|Fort Lauderdale–Hollywood International Airport
FLO|Florence Regional Airport
FNT|Bishop International Airport
FSD|Sioux Falls Regional Airport
FSM|Fort Smith Regional Airport
FWA|Fort Wayne International Airport
GCC|Northeast Wyoming Regional Airport
GCK|Garden City Regional Airport
GEG|Spokane International Airport
GFK|Grand Forks International Airport
GGG|East Texas Regional Airport
GJT|Grand Junction Regional Airport
GNV|Gainesville Regional Airport
GPT|Gulfport–Biloxi International Airport
GRB|Green Bay–Austin Straubel International Airport
GRI|Central Nebraska Regional Airport
GRK|Killeen–Fort Hood Regional Airport
GRR|Gerald R. Ford International Airport
GSO|Piedmont Triad International Airport
GSP|Greenville–Spartanburg International Airport
GST|Gustavus Airport
GTF|Great Falls International Airport
GTR|Golden Triangle Regional Airport
GUC|Gunnison–Crested Butte Regional Airport
HDN|Yampa Valley Airport
HGR|Hagerstown Regional Airport
HHH|Hilton Head Airport
HIB|Range Regional Airport
HLN|Helena Regional Airport
HNL|Daniel K. Inouye International Airport
HOB|Lea County Regional Airport
HOU|William P. Hobby Airport
HPN|Westchester County Airport
HRL|Valley International Airport
HSV|Huntsville International Airport
HTS|Tri-State Airport
HVN|Tweed–New Haven Airport
HYA|Cape Cod Gateway Airport
IAD|Washington Dulles International Airport
IAG|Niagara Falls International Airport
IAH|George Bush Intercontinental Airport
ICT|Wichita Dwight D. Eisenhower National Airport
IDA|Idaho Falls Regional Airport
ILG|Wilmington Airport (Delaware)
ILM|Wilmington International Airport
IMT|Ford Airport
IND|Indianapolis International Airport
INL|Falls International Airport
IPL|Imperial County Airport
ISN|Sloulin Field International Airport
ISP|Long Island MacArthur Airport
ITH|Ithaca Tompkins International Airport
ITO|Hilo International Airport
JAC|Jackson Hole Airport
JAN|Jackson–Medgar Wiley Evers International Airport
JAX|Jacksonville International Airport
JFK|John F. Kennedy International Airport
JHM|Kapalua Airport
JLN|Joplin Regional Airport
JMS|Jamestown Regional Airport
JNU|Juneau International Airport
KOA|Ellison Onizuka Kona International Airport at Keāhole
KTN|Ketchikan International Airport
LAN|Capital Region International Airport
LAR|Laramie Regional Airport
LAS|Harry Reid International Airport
LAW|Lawton–Fort Sill Regional Airport
LAX|Los Angeles International Airport
LBB|Lubbock Preston Smith International Airport
LBE|Arnold Palmer Regional Airport
LCH|Lake Charles Regional Airport
LEX|Blue Grass Airport
LFT|Lafayette Regional Airport
LGA|LaGuardia Airport
LGB|Long Beach Airport
LIH|Lihue Airport
LIT|Bill and Hillary Clinton National Airport
LNK|Lincoln Airport
LNY|Lanai Airport
LRD|Laredo International Airport
LSE|La Crosse Regional Airport
LWB|Greenbrier Valley Airport
LWS|Lewiston–Nez Perce County Airport
LYH|Lynchburg Regional Airport
MAF|Midland International Air and Space Port
MBS|MBS International Airport
MCI|Kansas City International Airport
MCO|Orlando International Airport
MDT|Harrisburg International Airport
MDW|Chicago Midway International Airport
MEI|Key Field Airport
MEM|Memphis International Airport
MFE|McAllen Miller International Airport
MFR|Rogue Valley International–Medford Airport
MGM|Montgomery Regional Airport
MHK|Manhattan Regional Airport
MHT|Manchester–Boston Regional Airport
MIA|Miami International Airport
MKE|Milwaukee Mitchell International Airport
MKG|Muskegon County Airport
MLB|Melbourne Orlando International Airport
MLI|Quad Cities International Airport
MLU|Monroe Regional Airport
MMH|Mammoth Yosemite Airport
MOB|Mobile Regional Airport
MOT|Minot International Airport
MQT|Marquette Sawyer Regional Airport
MRY|Monterey Regional Airport
MSN|Dane County Regional Airport
MSO|Missoula Montana Airport
MSP|Minneapolis–Saint Paul International Airport
MSY|Louis Armstrong New Orleans International Airport
MTJ|Montrose Regional Airport
MVY|Martha's Vineyard Airport
MYR|Myrtle Beach International Airport
OAJ|Albert J. Ellis Airport
OAK|Oakland San Francisco Bay Airport
OGG|Kahului Airport
OKC|Will Rogers World Airport
OMA|Eppley Airfield
OME|Nome Airport
ONT|Ontario International Airport
ORD|O'Hare International Airport
ORF|Norfolk International Airport
OTH|Southwest Oregon Regional Airport
OTZ|Ralph Wien Memorial Airport
PAE|Seattle Paine Field International Airport
PBG|Plattsburgh International Airport
PBI|Palm Beach International Airport
PDX|Portland International Airport
PGD|Punta Gorda Airport
PGV|Pitt–Greenville Airport
PHF|Newport News/Williamsburg International Airport
PHL|Philadelphia International Airport
PHX|Phoenix Sky Harbor International Airport
PIA|General Wayne A. Downing Peoria International Airport
PIB|Hattiesburg–Laurel Regional Airport
PIE|St. Pete–Clearwater International Airport
PIH|Pocatello Regional Airport
PIT|Pittsburgh International Airport
PLN|Pellston Regional Airport
PNS|Pensacola International Airport
PPG|Pago Pago International Airport
PSC|Tri-Cities Airport (Washington)
PSE|Mercedita International Airport
PSG|Petersburg James A. Johnson Airport
PSM|Portsmouth International Airport at Pease
PSP|Palm Springs International Airport
PUB|Pueblo Memorial Airport
PUW|Pullman–Moscow Regional Airport
PVD|Rhode Island T. F. Green International Airport
PWM|Portland International Jetport
RAP|Rapid City Regional Airport
RDD|Redding Regional Airport
RDM|Redmond Municipal Airport
RDU|Raleigh–Durham International Airport
RFD|Chicago Rockford International Airport
RHI|Rhinelander–Oneida County Airport
RIC|Richmond International Airport
RKS|Southwest Wyoming Regional Airport
RNO|Reno–Tahoe International Airport
ROA|Roanoke–Blacksburg Regional Airport
ROC|Frederick Douglass Greater Rochester International Airport
ROW|Roswell Air Center
RST|Rochester International Airport (Minnesota)
RSW|Southwest Florida International Airport
SAF|Santa Fe Regional Airport
SAN|San Diego International Airport
SAT|San Antonio International Airport
SAV|Savannah/Hilton Head International Airport
SBA|Santa Barbara Municipal Airport
SBN|South Bend International Airport
SBP|San Luis Obispo County Regional Airport
SCC|Deadhorse Airport
SCE|University Park Airport
SDF|Louisville Muhammad Ali International Airport
SEA|Seattle–Tacoma International Airport
SFB|Orlando Sanford International Airport
SFO|San Francisco International Airport
SGF|Springfield–Branson National Airport
SGU|St. George Regional Airport
SHD|Shenandoah Valley Regional Airport
SHR|Sheridan County Airport
SHV|Shreveport Regional Airport
SIT|Sitka Rocky Gutierrez Airport
SJC|San José Mineta International Airport
SJT|San Angelo Regional Airport
SJU|Luis Muñoz Marín International Airport
SLC|Salt Lake City International Airport
SLN|Salina Regional Airport
SMF|Sacramento International Airport
SMX|Santa Maria Public Airport
SNA|John Wayne Airport
SPI|Abraham Lincoln Capital Airport
SPS|Wichita Falls Regional Airport
SRQ|Sarasota Bradenton International Airport
STC|St. Cloud Regional Airport
STL|St. Louis Lambert International Airport
STS|Charles M. Schulz–Sonoma County Airport
STT|Cyril E. King Airport
STX|Henry E. Rohlsen Airport
SUN|Friedman Memorial Airport
SUX|Sioux Gateway Airport
SWF|New York Stewart International Airport
SYR|Syracuse Hancock International Airport
TLH|Tallahassee International Airport
TOL|Eugene F. Kranz Toledo Express Airport
TPA|Tampa International Airport
TRI|Tri-Cities Airport (Tennessee)
TTN|Trenton–Mercer Airport
TUL|Tulsa International Airport
TUP|Tupelo Regional Airport
TUS|Tucson International Airport
TVC|Cherry Capital Airport
TWF|Magic Valley Regional Airport
TXK|Texarkana Regional Airport
TYR|Tyler Pounds Regional Airport
TYS|McGhee Tyson Airport
USA|Concord–Padgett Regional Airport
VLD|Valdosta Regional Airport
VPS|Destin–Fort Walton Beach Airport
WRG|Wrangell Airport
WYS|Yellowstone Airport
XNA|Northwest Arkansas National Airport
YAK|Yakutat Airport
YKM|Yakima Air Terminal
YUM|Yuma International Airport
""".splitlines()
    if line.strip()
)


def carrier_full_name(code: object) -> str:
    """Return label penerbangan tanpa mengubah kode input model."""
    value = str(code).strip().upper()
    return CARRIER_NAMES.get(value, "")


def airport_full_name(code: object) -> str:
    """Nama tampilan; kosong bila kode tidak terpetakan (no guessing)."""
    value = str(code).strip().upper()
    return AIRPORT_NAMES.get(value, "")


def carrier_option(code: str) -> str:
    name = carrier_full_name(code)
    return f"{name} ({code})" if name else f"{code} (nama operator belum tersedia)"


st.set_page_config(
    page_title="ATL Flight Intelligence | Departure Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Tampilan dashboard operasional dark mode. Seluruh aset tersedia offline.
st.markdown(
    """
<style>
:root { color-scheme: dark; }
.stApp {
  --text-color: #e6edf5;
  --background-color: #0e151e;
  --secondary-background-color: #18232e;
  --primary-color: #699fc2;
  background: #0e151e;
  color: #e6edf5;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] { background-color: #0e151e; }
.block-container { max-width: 1140px; padding-top: 1.65rem; padding-bottom: 2.8rem; }
#MainMenu, footer { visibility: hidden; }

/* Identitas visual sederhana, seperti dashboard operasional. */
.atl-topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 17px; border-bottom: 1px solid #2d3a49;
  gap: 12px; flex-wrap: wrap;
}
.atl-identity { display: flex; align-items: center; gap: 12px; }
.atl-mark {
  display: flex; align-items: center; justify-content: center;
  width: 43px; height: 38px; background: #24394b; color: #eef4f8;
  border: 1px solid #3b5164; border-radius: 5px;
  font-weight: 750; font-size: .92rem; letter-spacing: -.035em;
}
.atl-name { color: #eef3f9; font-weight: 770; font-size: .95rem; }
.atl-name-sub { color: #a1b0c0; font-size: .75rem; margin-top: 2px; }
.atl-top-meta { color: #91a3b5; font-size: .75rem; text-align: right; }
.atl-intro { padding: 28px 0 13px; }
.atl-intro h1 {
  margin: 0 0 8px; color: #edf4fb !important;
  letter-spacing: -.033em; font-size: clamp(1.65rem, 2.7vw, 2.2rem);
  font-weight: 750; line-height: 1.17;
}
.atl-intro p { margin: 0; color: #acbbc9; font-size: .91rem; line-height: 1.58; }
.atl-meta-line {
  display: flex; flex-wrap: wrap; gap: 7px 18px;
  margin-top: 14px; color: #91a5b7; font-size: .76rem;
}
.atl-meta-line strong { color: #dce6f0; font-weight: 650; }
.atl-note {
  border-left: 3px solid #58778e; background: #17232e;
  color: #a6b8c7; font-size: .79rem; line-height: 1.55;
  padding: 10px 14px; margin: 14px 0 29px;
}
.atl-note strong { color: #dce6f0; }

/* Ruang, kartu, dan teks: tanpa gradient, glow, atau ornamen berlebihan. */
.atl-section-head { margin: 0 0 16px; }
.atl-kicker {
  color: #91a4b7; font-size: .72rem; font-weight: 700;
  letter-spacing: .025em; margin-bottom: 5px;
}
.atl-section-head h2 {
  color: #e6edf5; font-size: 1.11rem; font-weight: 730;
  margin: 0; letter-spacing: -.018em;
}
.atl-section-head p { font-size: .78rem; color: #9aabbc; margin: 5px 0 0; }
.atl-search-label {
  color: #a6b6c5; font-size: .74rem; margin: 4px 0 11px;
  border-top: 1px solid #2d3a49; padding-top: 14px;
}
.atl-route-card {
  background: #18232e; border: 1px solid #334152;
  border-radius: 8px; padding: 21px 24px 23px;
}
.atl-route-top {
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
  color: #9bafc1; font-size: .72rem; padding-bottom: 15px;
  border-bottom: 1px solid #334152;
}
.atl-record-id { color: #9fb2c3; font-variant-numeric: tabular-nums; }
.atl-route {
  display: grid; grid-template-columns: minmax(0,1fr) 60px minmax(0,1fr);
  align-items: center; gap: 12px; padding: 26px 0 22px;
}
.atl-airport-code {
  font-size: clamp(2.3rem, 4vw, 3.1rem); font-weight: 750;
  line-height: 1; color: #eaf2fa; letter-spacing: -.055em;
}
.atl-airport-name {
  margin-top: 7px; font-size: .78rem; color: #b3c2d0;
  line-height: 1.45; overflow-wrap: anywhere;
}
.atl-airport-role { margin-top: 8px; color: #91a3b4; font-size: .69rem; }
.atl-airport-end { text-align: right; }
.atl-route-arrow { color: #92aec5; font-size: 1.6rem; text-align: center; }
.atl-flight-row {
  display: flex; justify-content: space-between; align-items: center;
  gap: 10px; flex-wrap: wrap;
  border-top: 1px solid #334152; border-bottom: 1px solid #334152;
  padding: 11px 0; margin-bottom: 20px;
}
.atl-flight-row-label { color: #9cadbd; font-size: .76rem; }
.atl-flight-row-value { color: #e3edf7; font-weight: 760; font-size: 1rem; }
.atl-details { display: grid; grid-template-columns: repeat(3,minmax(0,1fr));
  column-gap: 17px; row-gap: 17px; }
.atl-label { color: #97a9ba; font-size: .72rem; margin-bottom: 4px; }
.atl-val { color: #e2e9f2; font-size: .84rem; font-weight: 610;
  overflow-wrap: anywhere; }

/* Hijau dan merah hanya untuk hasil klasifikasi. */
.atl-prediction-board {
  display: grid; grid-template-columns: 1.45fr 1fr 1fr;
  border: 1px solid #334152; border-radius: 8px;
  background: #18232e; overflow: hidden; margin-top: 2px;
}
.atl-prediction-status { padding: 24px; border-right: 1px solid #334152; }
.atl-prediction-metric { padding: 24px; border-right: 1px solid #334152; }
.atl-prediction-metric:last-child { border-right: none; }
.atl-result-title { color: #a6b7c7; font-size: .74rem; }
.atl-status-line { display: flex; align-items: center; gap: 10px; margin: 11px 0 8px; }
.atl-status-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.atl-status-word {
  font-size: clamp(1.85rem, 3.2vw, 2.65rem); font-weight: 760;
  letter-spacing: -.035em; line-height: 1.15;
}
.atl-prediction-board.ontime .atl-status-word { color: #5fdda1; }
.atl-prediction-board.delayed .atl-status-word { color: #ff8388; }
.atl-prediction-board.ontime .atl-status-dot { background: #5fdda1; }
.atl-prediction-board.delayed .atl-status-dot { background: #ff8388; }
.atl-result-detail { color: #9fb0c0; font-size: .72rem; line-height: 1.5; }
.atl-result-number {
  color: #e5edf5; font-size: clamp(1.35rem,2.5vw,1.95rem);
  margin: 14px 0 9px; font-weight: 735; letter-spacing: -.035em;
  font-variant-numeric: tabular-nums;
}
.atl-score-panel {
  background: #18232e; border: 1px solid #334152; border-radius: 8px;
  padding: 18px 22px; margin-top: 12px;
}
.atl-score-head {
  display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap;
  font-size: .76rem; color: #a6b7c7; margin-bottom: 14px;
}
.atl-score-head strong { color: #e6edf5; }
.atl-score-track { height: 8px; border-radius: 4px; background: #354456; position: relative; }
.atl-score-fill { height: 100%; border-radius: 4px; background: #5fdda1; }
.atl-score-fill.delayed { background: #ff8388; }
.atl-score-marker {
  height: 18px; width: 2px; background: #e1eaf3;
  position: absolute; top: -5px; box-shadow: 0 0 0 2px #18232e;
}
.atl-score-foot {
  display: flex; justify-content: space-between; gap: 10px;
  margin-top: 10px; font-size: .69rem; color: #96a8b9;
}
.atl-context { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 13px; }
.atl-context-item {
  border: 1px solid #334152; border-radius: 6px;
  padding: 10px 13px; font-size: .78rem; color: #adbdcb; background: #18232e;
}
.atl-context-item strong { color: #e5edf5; font-weight: 670; }
.atl-shap-note { font-size: .8rem; color: #a6b7c7; line-height: 1.6; margin: 4px 0 13px; }
.atl-footer {
  border-top: 1px solid #2d3a49; margin-top: 28px; padding-top: 15px;
  display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;
  font-size: .73rem; color: #91a4b5;
}

/* Widget Streamlit, termasuk dropdown yang muncul di atas halaman. */
[data-testid="stSelectbox"] label { color: #d2dee8 !important; font-size: .84rem; font-weight: 620; }
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  border: 1px solid #3a4b5c !important; border-radius: 6px;
  background: #18232e !important; color: #e6edf5 !important;
}
[data-testid="stSelectbox"] div[data-baseweb="select"] input { color: #e6edf5 !important; }
[data-testid="stSelectbox"] div[data-baseweb="select"] svg { fill: #b6c6d4 !important; }
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] ul,
[data-baseweb="popover"] [role="option"] {
  background: #1b2835 !important; color: #e6edf5 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"] {
  background: #2a4154 !important;
}
.stButton > button[kind="primary"] {
  background: #376d92; border: 1px solid #477fa5;
  color: #fff; border-radius: 6px; min-height: 44px; font-weight: 650;
}
.stButton > button[kind="primary"]:hover {
  background: #4381aa; border-color: #699fc2; color: #fff;
}
.stButton > button:focus-visible { outline: 2px solid #8fc7e8; outline-offset: 2px; }
[data-testid="stExpander"] {
  background: #18232e; border: 1px solid #334152; border-radius: 7px;
}
[data-testid="stExpander"] summary { color: #e0e9f2; }
[data-testid="stCaptionContainer"], .stCaption { color: #9fb0c0 !important; }
[data-testid="stMarkdownContainer"] > p { color: #bac8d5; }
[data-testid="stDataFrame"] { border: 1px solid #334152; border-radius: 6px; }
/* Grafik SHAP dari matplotlib dipertahankan warna aslinya supaya tetap terbaca. */
[data-testid="stPyplot"] img { border: 1px solid #334152; border-radius: 6px; }

@media(max-width:900px) {
  .atl-prediction-board { grid-template-columns: 1fr 1fr; }
  .atl-prediction-status { grid-column: 1 / -1; border-right: 0; border-bottom: 1px solid #334152; }
}
@media(max-width:650px) {
  .block-container { padding: 1rem 1rem 2rem; }
  .atl-intro { padding-top: 22px; }
  .atl-top-meta { text-align: left; }
  .atl-route-card { padding: 18px 16px; }
  .atl-route { grid-template-columns: minmax(0,1fr) 32px minmax(0,1fr); gap: 8px; }
  .atl-details { grid-template-columns: repeat(2,minmax(0,1fr)); gap: 14px; }
  .atl-airport-code { font-size: 2.3rem; }
  .atl-prediction-board { grid-template-columns: 1fr; }
  .atl-prediction-status { grid-column: auto; }
  .atl-prediction-metric {
    border-right: 0; border-bottom: 1px solid #334152; padding: 17px 22px;
  }
  .atl-prediction-metric:last-child { border-bottom: 0; }
  .atl-score-panel { padding: 16px; }
}
</style>
    """,
    unsafe_allow_html=True,
)


def safe(value: object) -> str:
    """Escape teks dataset sebelum dimasukkan ke HTML kartu."""
    return escape(str(value), quote=True)


def pretty_date(value: str) -> str:
    d = pd.Timestamp(value)
    return f"{d.day} {MONTHS_ID[d.month]} {d.year}"


def pretty_hhmm(value: object) -> str:
    hhmm = f"{int(value):04d}"
    return f"{hhmm[:2]}:{hhmm[2:]}"


def flight_number(row: pd.Series) -> str:
    """Nomor penerbangan hanya untuk identifikasi, bukan fitur tambahan."""
    if "op_carrier_fl_num" not in row.index or pd.isna(row["op_carrier_fl_num"]):
        return "—"
    raw = row["op_carrier_fl_num"]
    try:
        return str(int(float(raw)))
    except (ValueError, TypeError, OverflowError):
        return str(raw)


def section_head(number: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="atl-section-head"><div class="atl-kicker">{safe(number)}</div>'
        f'<h2>{safe(title)}</h2><p>{safe(description)}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_bundle(path: str) -> dict:
    # Joblib memakai pickle: hanya muat artefak milik sendiri yang dipercaya.
    bundle = joblib.load(path)
    verify_bundle(bundle)
    return bundle


@st.cache_data(show_spinner=False)
def load_flights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"op_unique_carrier": "string", "dest": "string"})
    required = {
        "observation_index", "fl_date", "crs_dep_time", "op_unique_carrier",
        "dest", "distance", "crs_elapsed_time",
    }
    if not required.issubset(df.columns):
        raise ValueError(f"CSV tidak memuat kolom {sorted(required - set(df.columns))}.")
    dates = pd.to_datetime(df["fl_date"], errors="raise")
    if not dates.between("2024-11-01", "2024-12-31").all():
        raise ValueError("CSV memuat data di luar periode pengujian Nov–Des 2024.")
    if df["observation_index"].duplicated().any():
        raise ValueError("Identitas observasi duplikat dalam CSV.")
    df = df.copy()
    df["fl_date"] = dates.dt.strftime("%Y-%m-%d")
    return df


st.markdown(
    """
<div class="atl-topbar">
  <div class="atl-identity">
    <div class="atl-mark">ATL</div>
    <div>
      <div class="atl-name">ATL Flight Intelligence</div>
      <div class="atl-name-sub">Prediksi keterlambatan keberangkatan</div>
    </div>
  </div>
  <div class="atl-top-meta">Data historis · November–Desember 2024</div>
</div>
<div class="atl-intro">
  <h1>Prediksi Keterlambatan Penerbangan</h1>
  <p>Pilih penerbangan dari Hartsfield–Jackson Atlanta International Airport (ATL) untuk melihat hasil prediksi dan penjelasan model.</p>
  <div class="atl-meta-line">
    <span>Model: <strong>LightGBM Baseline V1</strong></span>
    <span>Horizon: <strong>T−2 jam</strong></span>
    <span>Fitur: <strong>7</strong></span>
    <span>Interpretasi: <strong>SHAP</strong></span>
  </div>
</div>
<div class="atl-note">
  <strong>Catatan:</strong> Aplikasi ini menggunakan data historis, bukan informasi penerbangan langsung.
  Horizon T−2 jam menggunakan asumsi ketersediaan jadwal dua jam sebelum waktu keberangkatan.
</div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file() or not DATA_PATH.is_file():
    st.error("Model atau dataset belum tersedia. Ekspor artefak dari notebook dan tempatkan di folder models/ dan data/.")
    st.code(f"Model: {MODEL_PATH}\nData: {DATA_PATH}", language="text")
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
    df = load_flights(str(DATA_PATH))
except Exception as exc:
    st.error(f"Artefak tidak dapat dimuat: {exc}")
    st.stop()

left, right = st.columns([0.9, 1.55], gap="large")

with left:
    section_head("01 · Pencarian", "Pilih penerbangan", "Pilih tanggal, maskapai, lalu penerbangan.")
    st.markdown('<div class="atl-search-label">Keberangkatan dari Atlanta (ATL)</div>', unsafe_allow_html=True)
    dates = sorted(df["fl_date"].unique().tolist())
    chosen_date = st.selectbox("Tanggal penerbangan", dates, format_func=pretty_date)
    day = df.loc[df["fl_date"] == chosen_date]
    carriers = sorted(day["op_unique_carrier"].dropna().unique().tolist())
    carrier = st.selectbox("Maskapai (operating carrier)", carriers, format_func=carrier_option)
    matches = day.loc[day["op_unique_carrier"] == carrier]
    options = matches.index.tolist()

    def flight_name(i: int) -> str:
        r = df.loc[i]
        # Selectbox menampilkan teks biasa, jadi tidak perlu HTML escaping di sini.
        return (
            f"{r['op_unique_carrier']}{flight_number(r)}  ·  "
            f"ATL → {r['dest']} ({airport_full_name(r['dest']) or 'nama bandara belum tersedia'})"
            f"  ·  {pretty_hhmm(r['crs_dep_time'])}"
            f"  [#{int(r['observation_index'])}]"
        )

    chosen = st.selectbox("Penerbangan historis", options, format_func=flight_name)
    submit = st.button("Lihat hasil prediksi", type="primary", use_container_width=True)
    st.caption("Prediksi memakai model terpilih dan ambang keputusan hasil validasi.")

flight = df.loc[chosen]
flight_date = pretty_date(str(flight["fl_date"]))
schedule_display = pretty_hhmm(flight["crs_dep_time"])
carrier_code = str(flight["op_unique_carrier"]).strip().upper()
carrier_display = safe(carrier_code)
carrier_name_display = safe(carrier_full_name(carrier_code) or "Nama operator belum tersedia")
dest_display = safe(flight["dest"])
dest_name_display = safe(airport_full_name(flight["dest"]) or "Nama bandara belum tersedia")
origin_name_display = safe(airport_full_name("ATL"))
flight_num_display = safe(flight_number(flight))
observation_id = int(flight["observation_index"])

with right:
    section_head("02 · Informasi penerbangan", "Ringkasan penerbangan", "Jadwal dan detail rute yang dipilih.")
    st.markdown(
        f"""
<div class="atl-route-card">
  <div class="atl-route-top">
    <span>Rute penerbangan</span>
    <span class="atl-record-id">ID observasi #{observation_id}</span>
  </div>
  <div class="atl-route">
    <div>
      <div class="atl-airport-code">ATL</div>
      <div class="atl-airport-name">{origin_name_display}</div>
      <div class="atl-airport-role">Asal</div>
    </div>
    <div class="atl-route-arrow" aria-hidden="true">→</div>
    <div class="atl-airport-end">
      <div class="atl-airport-code">{dest_display}</div>
      <div class="atl-airport-name">{dest_name_display}</div>
      <div class="atl-airport-role">Tujuan</div>
    </div>
  </div>
  <div class="atl-flight-row">
    <span class="atl-flight-row-label">Nomor penerbangan</span>
    <span class="atl-flight-row-value">{carrier_display}{flight_num_display}</span>
  </div>
  <div class="atl-details">
    <div><div class="atl-label">Tanggal</div><div class="atl-val">{safe(flight_date)}</div></div>
    <div><div class="atl-label">Keberangkatan terjadwal</div><div class="atl-val">{safe(schedule_display)} (ATL)</div></div>
    <div><div class="atl-label">Maskapai pengoperasi</div><div class="atl-val">{carrier_name_display} ({carrier_display})</div></div>
    <div><div class="atl-label">Rute</div><div class="atl-val">ATL → {dest_display}</div></div>
    <div><div class="atl-label">Jarak</div><div class="atl-val">{float(flight['distance']):,.0f} mil</div></div>
    <div><div class="atl-label">Durasi terjadwal</div><div class="atl-val">{float(flight['crs_elapsed_time']):,.0f} menit</div></div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

if submit:
    try:
        # Perhitungan dan keputusan kelas tetap dikerjakan oleh modul proyek asli.
        result = predict(flight, bundle)
    except Exception as exc:
        st.error(f"Prediksi gagal: {exc}")
    else:
        st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
        section_head(
            "03 · Hasil prediksi",
            "Hasil prediksi",
            "Kategori ditentukan berdasarkan probabilitas Delayed dan threshold hasil validasi.",
        )
        probability = float(result["prob"])
        threshold = float(result["threshold"])
        delayed = bool(result["label"])
        category = "Delayed" if delayed else "On-Time"
        category_css = "delayed" if delayed else "ontime"

        st.markdown(
            f"""
<div class="atl-prediction-board {category_css}" role="status" aria-live="polite">
  <div class="atl-prediction-status">
    <div class="atl-result-title">Status prediksi</div>
    <div class="atl-status-line">
      <span class="atl-status-dot" aria-hidden="true"></span>
      <span class="atl-status-word">{safe(category)}</span>
    </div>
    <div class="atl-result-detail">Hasil model, bukan status penerbangan aktual</div>
  </div>
  <div class="atl-prediction-metric">
    <div class="atl-result-title">Probabilitas Delayed</div>
    <div class="atl-result-number">{probability:.2%}</div>
    <div class="atl-result-detail">Probabilitas untuk kelas keterlambatan</div>
  </div>
  <div class="atl-prediction-metric">
    <div class="atl-result-title">Threshold validasi</div>
    <div class="atl-result-number">{threshold:.4f}</div>
    <div class="atl-result-detail">Ambang keputusan model</div>
  </div>
</div>
<div class="atl-score-panel">
  <div class="atl-score-head">
    <span>Probabilitas keterlambatan</span>
    <span>Threshold: <strong>{threshold:.2%}</strong></span>
  </div>
  <div class="atl-score-track" aria-label="Probabilitas Delayed {probability:.2%}; ambang {threshold:.2%}">
    <div class="atl-score-fill {category_css}" style="width:{probability * 100:.3f}%"></div>
    <div class="atl-score-marker" style="left:{threshold * 100:.3f}%" title="Threshold validasi"></div>
  </div>
  <div class="atl-score-foot"><span>0% · Lebih rendah</span><span>100% · Lebih tinggi</span></div>
</div>
<div class="atl-context">
  <div class="atl-context-item">Jadwal: <strong>{safe(result['schedule'].strftime('%d %b %Y, %H:%M'))}</strong> (ATL)</div>
  <div class="atl-context-item">Batas T−2 jam: <strong>{safe(result['cutoff'].strftime('%d %b %Y, %H:%M'))}</strong> (ATL)</div>
</div>
            """,
            unsafe_allow_html=True,
        )

        if result["unknown"]:
            unknown_labels = ", ".join(f"{col}={value}" for col, value in result["unknown"])
            st.warning(
                "Kategori yang tidak ada dalam training ditangani sebagai nilai hilang "
                f"sesuai konfigurasi model: {unknown_labels}"
            )

        with st.expander("Lihat tujuh fitur yang digunakan model", expanded=False):
            X_view = result["X"].T.copy()
            X_view.columns = ["Nilai untuk model"]
            st.dataframe(X_view, use_container_width=True)
            st.caption("Atribut aktual penerbangan target tidak digunakan dalam perhitungan prediksi.")

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        section_head(
            "04 · Interpretasi model",
            "Penjelasan prediksi dengan SHAP",
            "Kontribusi masing-masing fitur untuk penerbangan terpilih.",
        )
        st.markdown(
            '<div class="atl-shap-note">'
            'Pada waterfall SHAP, <b>merah</b> menaikkan dan <b>biru</b> menurunkan skor model '
            'untuk kelas <b>Delayed</b>. Nilai berada pada skala <i>raw score/log-odds</i>, '
            'bukan perubahan probabilitas secara langsung.</div>',
            unsafe_allow_html=True,
        )
        try:
            fig, _raw = local_waterfall(bundle, result["X"])
            st.pyplot(fig, clear_figure=True)
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception as exc:
            st.error(f"Interpretasi SHAP tidak dapat ditampilkan: {exc}")

st.markdown(
    """
<div class="atl-footer">
  <span>ATL Flight Intelligence · Prototipe penelitian akademik</span>
  <span>Data historis 2024 · Tidak untuk keputusan operasional</span>
</div>
    """,
    unsafe_allow_html=True,
)
