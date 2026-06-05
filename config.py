#!/usr/bin/env python3
"""
MEET & GREET KPI - Configuration File (V4)
All algorithm-driving parameters and camera configurations
"""

from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parent


def _resolve_model_path() -> str:
    """Resolve the YOLO weights file from the workspace directory.

    This project now uses a single canonical weights file named
    `security_latest 2.pt`. Return its absolute path if present,
    otherwise raise FileNotFoundError.
    """
    expected_name = "best.pt" #"security_latest 2.pt"
    expected_path = BASE_DIR / expected_name
    if expected_path.exists():
        return str(expected_path)
    raise FileNotFoundError(f"No model weights file found. Expected: {expected_name}")

# ===================== CAMERA CONFIGURATIONS =====================
RTSP_CAMERAS = [

    #Telangana Cameras Below
    # index: 0
    # ── SOMAJIGUDA — rectangle zones (ratios → rect)
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.108.159:8001/Streaming/Channels/301",
        "camera_id": "GF-1-CAM-3",
        "site_id": "1",
        "site_name": "somajiguda",
        "zone_mode": "rect",
        "entry_zone_1_ratios": [
            [0.3719, 0.3547],
            [0.5400, 0.3547],
            [0.5400, 0.4105],
            [0.3719, 0.4105],
        ],
        "entry_zone_2_ratios": [
            [0.3100, 0.4802],
            [0.5875, 0.4802],
            [0.5875, 0.8070],
            [0.3100, 0.8070],
        ],
    },

    # index: 1
    # ── JUBILEE HILLS — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.121.130:8001/Streaming/Channels/301",
        "camera_id": "GF-2-CAM-3",
        "site_id": "2",
        "site_name": "jubilee_hills",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0048, 0.5954], [0.0301, 0.8921], [0.115, 0.8487], [0.0751, 0.5711]],
        "entry_zone_2_ratios": [[0.115, 0.5151], [0.3906, 0.3487], [0.5517, 0.5829], [0.2009, 0.9197]],
    },

    # index: 2
    # ── CHANDANAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.100.131:8001/Streaming/Channels/501",
        "camera_id": "GF-11-CAM-5",
        "site_id": "11",
        "site_name": "chandanagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.5493119266, 0.3590535973], [0.5447247706, 0.4503138580], [0.6628440367, 0.4847899565], [0.6628440367, 0.3996137132]],
        "entry_zone_2_ratios": [[0.6330275229, 0.9897633993], [0.1387614679, 0.9857073877], [0.4128440367, 0.5294060840], [0.6628440367, 0.6531144375]],
    },

    # index: 3
    # ── HIMAYATHNAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.97.83:8001/Streaming/Channels/201",
        "camera_id": "GF-6-CAM-2",
        "site_id": "6",
        "site_name": "himayatnagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1926605505, 0.5540265036], [0.2580275229, 0.6824668705], [0.3566513761, 0.5846075433], [0.2763761468, 0.4765545362]],
        "entry_zone_2_ratios": [[0.0389908257, 0.1931702345], [0.5172018349, 0.1401630989], [0.5699541284, 0.8394495413], [0.1364678899, 0.9719673802]],
    },

    # index: 4
    # ── KARMINAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@45.117.66.58:8001/Streaming/Channel/801",
        "camera_id": "GF-21-CAM-8",
        "site_id": "21",
        "site_name": "karimnagar-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.3016055046, 0.5415741188], [0.2970183486, 0.7971028489], [0.3922018349, 0.8052148720], [0.3795871560, 0.5172380493], [0.3807339450, 0.5152100435]],
        "entry_zone_2_ratios": [[0.4025229358, 0.0629647513], [0.4036697248, 0.9674553356], [0.9472477064, 0.9005311444], [0.9449541284, 0.0447126992]],
    },

    # index: 5
    # ── KHAMMAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.113.18:8001/Streaming/Channels/401",
        "camera_id": "GF-30-CAM-4",
        "site_id": "30",
        "site_name": "khammam-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.4780676606, 0.2545234455], [0.4690366972, 0.3318042813], [0.6232548863, 0.3463635155], [0.625, 0.2659913354]],
        "entry_zone_2_ratios": [[0.3590192644, 0.3552247519], [0.7270642202, 0.3888888889], [0.8887614679, 0.9719673802], [0.1915137615, 0.9332313965]],
    },

    # index: 6
    # ── KOKAPET — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.112.217:8001/Streaming/Channels/601",
        "camera_id": "GF-12-CAM-6",
        "site_id": "12",
        "site_name": "kokapet-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.002293578, 0.7220666345], [0.0814220183, 0.9958474167], [0.2110091743, 0.9958474167], [0.0091743119, 0.5253500724]],
        "entry_zone_2_ratios": [[0.1364678899, 0.4422018349], [0.4162844037, 0.9654273298], [0.9575688073, 0.6551424433], [0.5493119266, 0.2150651859]],
    },

    # index: 7
    # ── KOMPALLY — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.106.41:8001/Streaming/Channels/401",
        "camera_id": "GF-10-CAM-4",
        "site_id": "10",
        "site_name": "kompally",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1548165138, 0.6409464027], [0.2029816514, 0.8478029937], [0.2591743119, 0.8031868662], [0.2155963303, 0.5841622405]],
        "entry_zone_2_ratios": [[0.1662844037, 0.2475132786], [0.5665137615, 0.0568807339], [0.8126094571, 0.4365102774], [0.2889908257, 0.9045871560]],
    },
    # index: 8
    # ── MIRYALAGUDA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@154.210.236.163:8001/Streaming/Channel/3001",
        "camera_id": "GF-18-CAM-30",
        "site_id": "18",    
        "site_name": "miryalaguda-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6339754816, 0.6161397364], [0.7810858144, 0.6594985713], [0.7810858144, 0.6594985713], [0.7600700525, 0.7926721357], [0.6094570928, 0.7462162411]],
        "entry_zone_2_ratios": [[0.8178633975, 0.1484837312], [0.7583187391, 1.0], [0.1541155867, 0.9908839524], [0.6147110333, 0.1391925523]],
    },
    # index: 9
    # ── NIZAMABAD — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@43.249.216.149:8001/Streaming/Channels/401",
        "camera_id": "GF-31-CAM-4",
        "site_id": "31",
        "site_name": "nizamabad",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.7259174312, 0.3976287012], [0.7568807339, 0.5200035871], [0.8107798165, 0.5035300448], [0.7763761468, 0.3835085221]],
        "entry_zone_2_ratios": [[0.5779816514, 0.0587444019], [0.9724770642, 0.1058116657], [0.9759174312, 0.9247820557], [0.5470183486, 0.9271354189]],
    },

    # index: 10
    # ── WARANGAL — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.105.66:8001/Streaming/Channels/701",
        "camera_id": "GF-15-CAM-7",
        "site_id": "15",
        "site_name": "warangal-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6681934528, 0.2192509767], [0.7375724101, 0.2264358525], [0.7658628587, 0.3030745281], [0.6847635727, 0.2991827204]],
        "entry_zone_2_ratios": [[0.3747810858, 0.3240902899], [0.7968476357, 0.3272037361], [0.9141856392, 0.8471492508], [0.3345008757, 0.9031912824]],
    },


    # Andhra Pradesh Cameras Below

    # index: 11
    # ── VIZAG — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.103:8002/Streaming/Channels/401?rtsp_transport=tcp&fflags=discardcorrupt&flags=low_delay&fflags=nobuffer",
        "camera_id": "FF-3-CAM-4",
        "site_id": "3",
        "site_name": "vizag",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6552, 0.2097], [0.6211, 0.2463], [0.6982, 0.3574], [0.7305, 0.3019]],
        "entry_zone_2_ratios": [[0.576, 0.2861], [0.3758, 0.4981], [0.7401, 0.906], [0.9109, 0.5537]],
    },

    # index: 12
    # ── VIJAYAWADA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.76.17:8001/Streaming/Channels/401?rtsp_transport=tcp&fflags=discardcorrupt&flags=low_delay&fflags=nobuffer",
        "camera_id": "GF-4-CAM-4",
        "site_id": "4",
        "site_name": "vijayawada",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.7031, 0.3227], [0.6669, 0.3782], [0.8427, 0.5], [0.8573, 0.4287]],
        "entry_zone_2_ratios": [[0.5867, 0.3833], [0.3289, 0.6958], [0.7557, 0.9861], [0.8672, 0.538]],
    },

    # index: 13
    # ── ANANTHAPUR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.248.210.164:8001/Streaming/Channels/301",
        "camera_id": "GF-26-CAM-3",
        "site_id": "26",
        "site_name": "ananthapur-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.5917431193, 0.4036697248], [0.5791284404, 0.5131820377], [0.7041284404, 0.5638821825], [0.7155963303, 0.4523418638]],
        "entry_zone_2_ratios": [[0.3520642202, 0.4280057943], [0.7110091743, 0.5922742636], [0.4323394495, 1.0], [0.0722477064, 0.9877353935]],
    },

    # index: 14
    # ── BHIMAVARAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.94.133:8001/Streaming/Channels/301",
        "camera_id": "GF-25-CAM-3",
        "site_id": "25",
        "site_name": "bhimavaram-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2626146789, 0.5922742636], [0.3291284404, 0.5537421535], [0.3314220183, 0.6024142926], [0.2683486239, 0.6450024143]],
        "entry_zone_2_ratios": [[0.6594036697, 0.9836793819], [0.3291284404, 0.3002414293], [0.1456422018, 0.3773056494], [0.126146789, 0.9877353935]],
    },

    # index: 15
    # ── ELURU — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.205.164.51:8001/Streaming/Channels/1201",
        "camera_id": "GF-13-CAM-12",
        "site_id": "13",
        "site_name": "eluru-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6685779817, 0.3752776437], [0.754587156, 0.4949299855], [0.7305045872, 0.5760502173], [0.6433486239, 0.4482858522]],
        "entry_zone_2_ratios": [[0.7201834862, 0.1724770642], [0.8658256881, 0.332689522], [0.6525229358, 0.9796233704], [0.1158256881, 0.9857073877], [0.121559633, 0.1968131338]],
    },

    # index: 16
    # ── GUNTUR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.205.164.154:8001/Streaming/Channels/601",
        "camera_id": "FF-19-CAM-6",
        "site_id": "19",
        "site_name": "guntur-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.79249155, 0.5104887601], [0.8546595847, 0.5941842374], [0.8348623853, 0.6600407747], [0.753440367, 0.5560652396]],
        "entry_zone_2_ratios": [[0.5917431193, 0.4031600408], [0.8337155963, 0.6906218145], [0.6548165138, 0.872069317], [0.4667431193, 1.0], [0.1834862385, 0.6641182467], [0.0928899083, 0.50509684], [0.245412844, 0.3644240571], [0.3830275229, 0.498980632]],
    },

    # index: 17
    # ── KADAPA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.171.190.6:8001/Streaming/Channels/301",
        "camera_id": "GF-17-CAM-3",
        "site_id": "17",
        "site_name": "kadapa-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0928196147, 0.6977038334], [0.1444444444, 0.9905349794], [0.357267951, 0.9903677758], [0.2407407407, 0.5757201646]],
        "entry_zone_2_ratios": [[0.2679509632, 0.5171239541], [0.4676007005, 0.9810274372], [0.5814360771, 0.8782837128], [0.8563922942, 0.4424012454], [0.5936952715, 0.2836154894]],
    },

    # index: 18
    # ── KAKINADA — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.145.50:8001/Streaming/Channels/301",
        "camera_id": "GF-22-CAM-3",
        "site_id": "22",
        "site_name": "kakinada-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.8130733945, 0.3419979613], [0.8612385321, 0.378695209], [0.8476357268, 0.4828760459], [0.7705779335, 0.4081533372]],
        "entry_zone_2_ratios": [[0.6865148862, 0.3521113057], [0.8616462347, 0.5264642927], [0.7670753065, 0.8284685737], [0.534150613, 0.4953298307]],
    },

    # index: 19
    # ── RAJAHMUNDRY — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.101:8001/Streaming/Channels/401",
        "camera_id": "GF-14-CAM-4",
        "site_id": "14",
        "site_name": "rajahmundry-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6690017513, 0.5541985436], [0.7267950963, 0.619236796], [0.7092819615, 0.789575076], [0.5905963303, 0.6368903911]],
        "entry_zone_2_ratios": [[0.0437828371, 0.7369250622], [0.5131348511, 0.6006544382], [0.6672504378, 0.9753986543], [0.0560420315, 0.9784957139]],
    },

    # index: 20
    # ── KURNOOL — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.206.112.148:8001/Streaming/Channels/501",
        "camera_id": "GF-16-CAM-5",
        "site_id": "16",
        "site_name": "kurnool-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.4334862385, 0.3184934814], [0.5183486239, 0.322549493], [0.5206422018, 0.401641719], [0.3635321101, 0.3996137132]],
        "entry_zone_2_ratios": [[0.0447247706, 0.1440849831], [0.0229357798, 0.9410912603], [0.7706422018, 0.9370352487], [0.6983944954, 0.1582810237]],
    },

    # index: 21
    # ── SRIKAKULAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.130.104:8001/Streaming/Channels/401",
        "camera_id": "GF-20-CAM-4",
        "site_id": "20",
        "site_name": "srikakulam-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.8841743119, 0.7089704383], [0.8245412844, 0.9006116208], [0.7087155963, 0.9108053007], [0.748853211, 0.7130479103]],
        "entry_zone_2_ratios": [[0.7293577982, 0.2849133537], [0.5435779817, 0.9638124363], [0.127293578, 0.378695209], [0.3016055046, 0.1156982671]],
    },

    # index: 22
    # ── TIRUPATHI — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@202.83.31.24:8001/Streaming/Channels/401",
        "camera_id": "GF-23-CAM-4",
        "site_id": "23",
        "site_name": "tirupathi-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6116125556, 0.3337724091], [0.6776236023, 0.3377640068], [0.6978310656, 0.459906896], [0.6017333513, 0.4519237006]],
        "entry_zone_2_ratios": [[0.4426046808, 0.5012471475], [0.7355516637, 0.5097383644], [0.7705779335, 0.9710944825], [0.3391179748, 0.9682640768]],
    },

    # index: 23
    # ── ONGOLE — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.81.163:8001/Streaming/Channels/301",
        "camera_id": "GF-27-CAM-3",
        "site_id": "27",
        "site_name": "ongole-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.4724770642, 0.9917914051], [0.622706422, 0.7281506519], [0.7224770642, 0.8194109126], [0.6410550459, 0.9938194109]],
        "entry_zone_2_ratios": [[0.0504587156, 0.2089811685], [0.247706422, 0.82549493], [0.501146789, 0.5354901014], [0.1880733945, 0.0873008209]],
    },

    # index: 24
    # ── VIZIANAGARAM — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.52.2:8001/Streaming/Channels/301",
        "camera_id": "GF-29-CAM-3",
        "site_id": "29",
        "site_name": "vizianagaram-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1182136602, 0.7201060518], [0.2006713368, 0.9575063242], [0.2881706735, 0.8465566346], [0.1958286897, 0.6257849953]],
        "entry_zone_2_ratios": [[0.4012099984, 0.2012241504], [0.4553415061, 0.2776451025], [0.4187231333, 0.6342762122], [0.5190256329, 0.8239133896], [0.3773284509, 0.9682640768], [0.1671708327, 0.4616214686]],
    },

    # index: 25
    # ── ANAKAPALLI — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@175.101.133.45:8001/Streaming/Channels/401",
        "camera_id": "GF-28-CAM-4",
        "site_id": "28",
        "site_name": "anakapalli-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.003440367, 0.5214171455], [0.002293578, 0.8049393567], [0.0894495413, 0.7600653376], [0.0791284404, 0.4989801359]],
        "entry_zone_2_ratios": [[0.0802752294, 0.0808358676], [0.4243119266, 0.0461604892], [0.5745412844, 0.3460005256], [0.1536697248, 0.7988201723]],
    },

    # index: 26
    # ── NELLORE — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@202.83.31.18:8001/Streaming/Channels/301",
        "camera_id": "GF-24-CAM-3",
        "site_id": "24",
        "site_name": "nellore-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.1341743119, 0.7334352701], [0.2069206223, 0.9916677747], [0.2812126047, 0.9269600674], [0.2049262066, 0.6681292381]],
        "entry_zone_2_ratios": [[0.1823394495, 0.378695209], [0.2832568807, 0.7171253823], [0.3394495413, 0.9148827727], [0.4300458716, 0.8455657492], [0.3646788991, 0.5479102956], [0.5527522936, 0.3419979613], [0.4552752294, 0.2074413863]],
    },

    # Karnataka Cameras Below

    # index: 27
    # ── JAYANAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.37.109:8001/Streaming/Channels/401",
        "camera_id": "GF-5-CAM-4",
        "site_id": "5",
        "site_name": "jayanagar",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2461, 0.6806], [0.2977, 0.762], [0.1542, 0.9287], [0.1268, 0.8019]],
        "entry_zone_2_ratios": [[0.1055, 0.9806], [0.5185, 0.5569], [0.7508, 0.7532], [0.4081, 0.9944]],
    },

    # index: 28
    # ── UB CITY BANGLORE — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.89.106:8001/Streaming/Channels/401",
        "camera_id": "GF-34-CAM-4",
        "site_id": "34",
        "site_name": "ub-city-banglore-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.6639908257, 0.5273780782], [0.7844036697, 0.6287783679], [0.7626146789, 0.7646547562], [0.5791284404, 0.5922742636]],
        "entry_zone_2_ratios": [[0.5802752294, 0.9917914051], [0.5263761468, 0.1887011106], [0.0080275229, 0.2353452438], [0.0103211009, 0.9816513761]],
    },

    # index: 29
    # ── BELLARY — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@103.199.209.58:8001/Streaming/Channels/1001",
        "camera_id": "GF-35-CAM-10",
        "site_id": "35",
        "site_name": "bellary-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.253440367, 0.2563710499], [0.2901376147, 0.2665647299], [0.2993119266, 0.3644240571], [0.2041284404, 0.3073394495]],
        "entry_zone_2_ratios": [[0.0126146789, 0.3134556575], [0.5607798165, 0.3053007136], [0.5814220183, 0.9678899083], [0.0424311927, 0.9658511723]],
    },

    # index: 30
    # ── MARATHAHALLI BANGLORE — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@106.51.66.143:8001/Streaming/Channels/1301",
        "camera_id": "GF-37-CAM-13",
        "site_id": "37",
        "site_name": "marathahalli-banglore-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.0412844037, 0.6804281346], [0.0550458716, 0.7762487258], [0.0894495413, 0.876146789], [0.1651376147, 0.7986748216], [0.0940366972, 0.5948012232]],
        "entry_zone_2_ratios": [[0.0584862385, 0.0565749235], [0.5676605505, 0.0280326198], [0.6605504587, 0.6845056065], [0.2419724771, 0.9291539246]],
    },

    # Tamil Nadu Cameras Below

    # index: 31
    # ── COIMBATORE — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@183.82.251.16:8001/Streaming/Channels/601",
        "camera_id": "GF-40-CAM-6",
        "site_id": "40",
        "site_name": "coimbatore-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.2856, 0.4806], [0.3452, 0.4419], [0.4151, 0.5418], [0.2856, 0.6315]],
        "entry_zone_2_ratios": [[0.6778, 0.2462], [0.7122, 0.9659], [0.2271, 0.9842], [0.2305, 0.1789]],
    },

    # index: 32
    # ── CHENNAI ANNANAGAR — polygon zones
    {
        "rtsp_url": "rtsp://Bluecloud:User%401964@49.207.187.203:8001/Streaming/Channels/1101",
        "camera_id": "GF-41-CAM-11",
        "site_id": "41",
        "site_name": "chennai-annanagar-store",
        "zone_mode": "poly",
        "entry_zone_1_ratios": [[0.4048, 0.1422], [0.4025, 0.2095], [0.4713, 0.2095], [0.4667, 0.1422]],
        "entry_zone_2_ratios": [[0.3062, 0.0219], [0.5734, 0.028], [0.7695, 0.6091], [0.1938, 0.6233]],
    },
]

# Ensure all RTSP URLs include recommended FFmpeg/OpenCV capture options
# By default append a conservative set of options that improves stability:
#   rtsp_transport=tcp          -> use TCP
#   fflags=discardcorrupt+genpts -> drop corrupt frames + regenerate PTS
#   flags=low_delay             -> reduce latency
#   reorder_queue_size=0        -> avoid internal reorder buffering
# Do NOT add `fflags=nobuffer` by default (can cause visible glitches on WAN).
# If a URL already contains ffmpeg/rtsp flags (e.g. contains 'fflags=',
# 'rtsp_transport=' or 'flags='), we assume it's intentionally configured
# and we will NOT append the defaults.
RTSP_SUFFIX = (
    "rtsp_transport=tcp"
    "&fflags=discardcorrupt+genpts"
    "&flags=low_delay"
    "&reorder_queue_size=0"
)

for cam in RTSP_CAMERAS:
    url = cam.get("rtsp_url")
    if not url:
        continue

    lower = url.lower()

    if any(
        k in lower
        for k in (
            "fflags=",
            "rtsp_transport=",
            "flags=",
            "genpts",
            "reorder_queue_size",
        )
    ):
        continue

    cam["rtsp_url"] = (
        url + "&" + RTSP_SUFFIX
        if "?" in url
        else url + "?" + RTSP_SUFFIX
    )

# ===================== BUSINESS HOURS & TIMEZONE =====================
IST = ZoneInfo("Asia/Kolkata")
BUSINESS_START = dtime(10, 0)  # 10:00 AM
BUSINESS_END = dtime(21, 0)    # 09:00 PM

# ===================== ENTRY ZONE (Zone-1) Parameters =====================
ENTRY_MARGIN = 5
ENTRY_DEBOUNCE_FRAMES = 3

# Time-based entry confirmation (alternative to frame-count debouncing)
# If `USE_TIME_BASED_ENTRY` is True, the system will require the customer
# to remain in the entry zone for at least `ENTRY_DEBOUNCE_SEC` seconds
# before starting a session. This avoids relying purely on frame counts
# and can be more robust on low-framerate or lossy streams.
USE_TIME_BASED_ENTRY = True
ENTRY_DEBOUNCE_SEC = 0.5

# ===================== GREET ZONE (Zone-2) Confirmation Parameters =====================
RECT2_CONFIRM_FRAMES = 2
RECT2_ABSENCE_ABORT_SEC = 20

# Time-based greet-zone confirmation (alternative to frame-count debouncing)
# If `USE_TIME_BASED_RECT2` is True, the system will require the customer
# + staff to both be present inside the greet zone for at least
# `RECT2_CONFIRM_SEC` seconds before marking the greet zone as confirmed.
USE_TIME_BASED_RECT2 = True
RECT2_CONFIRM_SEC = 0.5

# ===================== GREET COEXISTENCE Parameters =====================
# Coexistence: When customer + staff are both in zone 2 for ≥2 seconds, save event.
# Gap tolerance: If they separate for >5s, restart the coexistence timer.
GREET_GAP_TOLERANCE = 5.0   # Max absence before restarting coexistence timer (seconds)
SESSION_MAX_SEC = 40        # Max session duration before auto-abort (seconds)

# ===================== COOLDOWN & RECOVERY =====================
COOLDOWN_SEC = 30

# After a greet is saved, use a short post-save cooldown before allowing
# the next coexistence window to start. This prevents immediate duplicate
# saves while allowing a fast restart when a new customer arrives.
POST_SAVE_COOLDOWN_SEC = 2.0
# ===================== DETECTION & FILTERING =====================
CONF_THRESHOLD = 0.50
MIN_W, MIN_H = 40, 100
CUSTOMER_LABEL = "customers"
GREEN_STAFF_LABELS = {
    "sec1",
    "sec2",
    "sec3",
    "sec4",
    "sec5",
    "sec6",
    "sec7",
    "sec8",
}
# Per-label confidence thresholds to reduce misclassification
# Detections classified as `CUSTOMER_LABEL` must meet this confidence
# to be considered a customer in downstream logic.
# Per-label thresholds removed — using the single global `CONF_THRESHOLD`
# for both customer and staff detections to keep behavior consistent.

# ===================== MODEL =====================
MODEL_PATH = _resolve_model_path()

# ===================== FRAME PROCESSING =====================
MAX_FPS = 25
FRAME_INTERVAL = 1.0 / MAX_FPS
PREVIEW_WINDOW_W = 1280
PREVIEW_WINDOW_H = 720

# ===================== INFERENCE WORKERS =====================
# Number of parallel GPU inference workers. Each worker loads its own model
# instance and owns an interleaved slice of cameras (cams[i::N]). More workers
# overlap CPU pre/post-processing with GPU compute and shorten each worker's
# round-robin rotation (faster per-camera revisit). VRAM budget is split across
# workers automatically. 2 is a good default for a single shared GPU.
NUM_INFERENCE_WORKERS = 1

# ===================== DISPLAY & HEADLESS MODE =====================
HEADLESS = False
PANEL_W = 300

# ===================== DEVICE CONFIGURATION =====================
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = "cpu"
except Exception:
    DEVICE = "cpu"

# ===================== OUTPUT & LOGGING =====================
# Use a repository-relative output directory by default
OUTPUT_BASE_DIR = "meet_and_greet"
LOG_DIR = "logs"
LOG_FILE = "meet_greet_log.txt"
LOG_BACKUP_COUNT = 5
