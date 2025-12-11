import numpy as np

"""
Differential power functions for gravitational atom processes.

Automatically converted from Mathematica using MathematicaToPythonConverter.py
"""


# Annihilation functions
def ann_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 2p annihilation.
    """
    return ( 1024 * G_N * alpha**20 * (-8 + alpha**2)**2 * (832 - 96 * alpha**2 + 13 * alpha**4)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)))/( 1225 * np.pi * r_g**4 * (64 + alpha**4)**8)



def ann_3p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 3p annihilation.
    """
    return 1/(1225 * np.pi * r_g**4 * (324 + alpha**4)**12) * 82944 * G_N * alpha**20 * (-18 + alpha**2)**2 * (884317824 - 257926032 * alpha**2 + 35061984 * alpha**4 - 2525256 * alpha**6 + 108216 * alpha**8 - 2457 * alpha**10 + 26 * alpha**12)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta))



def ann_3d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 3d annihilation.
    """
    return 1/(121 * np.pi * r_g**4 * (324 + alpha**4)**12) * 5184 * G_N * alpha**20 * (-18 + alpha**2)**6 * (1154736 - 326592 * alpha**2 + 24408 * alpha**4 - 1008 * alpha**6 + 11 * alpha**8)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**4



def ann_4p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 4p annihilation.
    """
    return 1/(1225 * np.pi * r_g**4 * (1024 + alpha**4)**16) * 65536 * G_N * alpha**20 * (-32 + alpha**2)**2 * (248823879412219904 - 87609086501191680 * alpha**2 + 14191396579704832 * alpha**4 - 1316115418447872 * alpha**6 + 76879914598400 * alpha**8 - 2944602734592 * alpha**10 + 75078041600 * alpha**12 - 1255145472 * alpha**14 + 13216768 * alpha**16 - 79680 * alpha**18 + 221 * alpha**20)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta))



def ann_4d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 4d annihilation.
    """
    return 1/(1089 * np.pi * r_g**4 * (1024 + alpha**4)**16) * 65536 * G_N * alpha**20 * (-32 + alpha**2)**6 * (108851651149824 - 45870250721280 * alpha**2 + 6685116596224 * alpha**4 - 488720302080 * alpha**6 + 19895681024 * alpha**8 - 477265920 * alpha**10 + 6375424 * alpha**12 - 42720 * alpha**14 + 99 * alpha**16)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**4



def ann_4f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 4f annihilation.
    """
    return 1/(169 * np.pi * r_g**4 * (1024 + alpha**4)**16) * 1677721600 * G_N * alpha**24 * (-32 + alpha**2)**10 * (13631488 - 1900544 * alpha**2 + 83968 * alpha**4 - 1856 * alpha**6 + 13 * alpha**8)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**8



def ann_5p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5p annihilation.
    """
    return 1/(49 * np.pi * r_g**4 * (2500 + alpha**4)**20) * 25600 * G_N * alpha**20 * (-50 + alpha**2)**2 * (206298828125000000000000000 - 78350830078125000000000000 * alpha**2 + 13802246093750000000000000 * alpha**4 - 1433612304687500000000000 * alpha**6 + 96974882812500000000000 * alpha**8 - 4499354492187500000000 * alpha**10 + 147532484375000000000 * alpha**12 - 3473316562500000000 * alpha**14 + 59012993750000000 * alpha**16 - 719896718750000 * alpha**18 + 6206392500000 * alpha**20 - 36700475000 * alpha**22 + 141335000 * alpha**24 - 320925 * alpha**26 + 338 * alpha**28)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta))



def ann_5d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5d annihilation.
    """
    return 1/(53361 * np.pi * r_g**4 * (2500 + alpha**4)**20) * 40000 * G_N * alpha**20 * (-50 + alpha**2)**6 * (1184326171875000000000000 - 575039062500000000000000 * alpha**2 + 104747070312500000000000 * alpha**4 - 10113703125000000000000 * alpha**6 + 586519882812500000000 * alpha**8 - 21756412500000000000 * alpha**10 + 532616437500000000 * alpha**12 - 8702565000000000 * alpha**14 + 93843181250000 * alpha**16 - 647277000000 * alpha**18 + 2681525000 * alpha**20 - 5888400 * alpha**22 + 4851 * alpha**24)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**4



def ann_5f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5f annihilation.
    """
    return 1/(169 * np.pi * r_g**4 * (2500 + alpha**4)**20) * 640000000000 * G_N * alpha**24 * (-50 + alpha**2)**10 * (507812500000000 - 125859375000000 * alpha**2 + 11629687500000 * alpha**4 - 543031250000 * alpha**6 + 14181250000 * alpha**8 - 217212500 * alpha**10 + 1860750 * alpha**12 - 8055 * alpha**14 + 13 * alpha**16)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**8



def ann_5g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5g annihilation.
    """
    return 1/(104329 * np.pi * r_g**4 * (2500 + alpha**4)**20) * 196000000000000 * G_N * alpha**28 * (-50 + alpha**2)**14 * (6056250000 - 518000000 * alpha**2 + 14925000 * alpha**4 - 207200 * alpha**6 + 969 * alpha**8)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**12



def ann_6p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6p annihilation.
    """
    return 1/(1225 * np.pi * r_g**4 * (5184 + alpha**4)**24) * 82944 * G_N * alpha**20 * (-72 + alpha**2)**2 * (9103912021922318433376259924702527488 - 3593886663440772671328367136892518400 * alpha**2 + 662972050588201180262504382754455552 * alpha**4 - 73285532442356471037735033284591616 * alpha**6 + 5377704158430110469928110834843648 * alpha**8 - 276990792666756940141197847953408 * alpha**10 + 10366376376306150698120009220096 * alpha**12 - 288252626507140995502571520000 * alpha**14 + 6040145303743698376511717376 * alpha**16 - 96134845999624326387597312 * alpha**18 + 1165151486061670211518464 * alpha**20 - 10726135710005329920000 * alpha**22 + 74410118182704513024 * alpha**24 - 383535317427683328 * alpha**26 + 1436388576509952 * alpha**28 - 3775967870976 * alpha**30 + 6589315008 * alpha**32 - 6890400 * alpha**34 + 3367 * alpha**36)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta))



def ann_6d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6d annihilation.
    """
    return 1/(5929 * np.pi * r_g**4 * (5184 + alpha**4)**24) * 5308416 * G_N * alpha**20 * (-72 + alpha**2)**6 * (281130981016280109985775845638144 - 146295613317454566614170872053760 * alpha**2 + 29640849133826270414615806476288 * alpha**4 - 3283637458482419052888320901120 * alpha**6 + 226616127044185183895433510912 * alpha**8 - 10429267235554295415161487360 * alpha**10 + 333026203433624342328508416 * alpha**12 - 7556413067307841801420800 * alpha**14 + 123430775076032065044480 * alpha**16 - 1457641409588704051200 * alpha**18 + 12392200190163419136 * alpha**20 - 74861550399651840 * alpha**22 + 313784033697792 * alpha**24 - 877061882880 * alpha**26 + 1527216768 * alpha**28 - 1454040 * alpha**30 + 539 * alpha**32)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**4



def ann_6f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6f annihilation.
    """
    return 1/(169 * np.pi * r_g**4 * (5184 + alpha**4)**24) * 55725627801600 * G_N * alpha**24 * (-72 + alpha**2)**10 * (252309329502949456478208 - 77394029847533404028928 * alpha**2 + 9512156456955452325888 * alpha**4 - 631472186255996878848 * alpha**6 + 25379778003065634816 * alpha**8 - 653945443310370816 * alpha**10 + 11121483446747136 * alpha**12 - 126146883354624 * alpha**14 + 944404033536 * alpha**16 - 4532723712 * alpha**18 + 13171008 * alpha**20 - 20672 * alpha**22 + 13 * alpha**24)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**8



def ann_6g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6g annihilation.
    """
    return 1/(104329 * np.pi * r_g**4 * (5184 + alpha**4)**24) * 62912004762894336 * G_N * alpha**28 * (-72 + alpha**2)**14 * (1166359680138608640 - 196345906021269504 * alpha**2 + 12585633038991360 * alpha**4 - 408352756727808 * alpha**6 + 7417452994560 * alpha**8 - 78771750912 * alpha**10 + 468322560 * alpha**12 - 1409376 * alpha**14 + 1615 * alpha**16)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**12



def ann_6h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6h annihilation.
    """
    return 1/(529 * np.pi * r_g**4 * (5184 + alpha**4)**24) * 106493333123540975616 * G_N * alpha**32 * (-72 + alpha**2)**18 * (4326690816 - 251942400 * alpha**2 + 5090688 * alpha**4 - 48600 * alpha**6 + 161 * alpha**8)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**16




# Transition functions
def trans_3p_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 3p 2p transition.
    """
    return (31701690482688 * G_N * alpha**12 * (1451520 + 71280 * alpha**2 + 617 * alpha**4 + 175 * alpha**2 * (144 + alpha**2) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(11962890625 * np.pi * r_g**4 * (144 + alpha**2)**10)



def trans_4p_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 4p 2p transition.
    """
    return (4294967296 * G_N * alpha**12 * (18350080 + 2191360 * alpha**2 + 104448 * alpha**4 + 1487 * alpha**6 + 105 * alpha**2 * (4096 + 384 * alpha**2 + 5 * alpha**4) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(361675125 * np.pi * r_g**4 * (64 + alpha**2)**12)



def trans_4p_3p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 4p 3p transition.
    """
    return (64925062108545024 * G_N * alpha**12 * (3 * (413516263587840 + 6909328097280 * alpha**2 + 11031220224 * alpha**4 + 8950848 * alpha**6 + 152929 * alpha**8) + 245 * alpha**2 * (30767579136 - 50761728 * alpha**2 - 39744 * alpha**4 + 245 * alpha**6) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(4154116321200125 * np.pi * r_g**4 * (576 + alpha**2)**14)



def trans_4d_3d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 4d 3d transition.
    """
    return (28047626830891450368 * G_N * alpha**12 * (67806393532416 + 1196814237696 * alpha**2 + 578672640 * alpha**4 - 6002784 * alpha**6 - 11319 * alpha**8 - 7 * alpha**2 * (-64465403904 - 292737024 * alpha**2 - 556416 * alpha**4 + 539 * alpha**6) * np.cos(2 * theta) + 776160 * alpha**4 * (576 + alpha**2) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(4021184598921721 * np.pi * r_g**4 * (576 + alpha**2)**14)



def trans_5p_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5p 2p transition.
    """
    return (275188285440000000 * G_N * alpha**12 * (12544000000000 + 2363200000000 * alpha**2 + 192422880000 * alpha**4 + 7708143600 * alpha**6 + 114652503 * alpha**8 + 3675 * alpha**2 * (89600000 + 15264000 * alpha**2 + 827280 * alpha**4 + 11907 * alpha**6) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(33232930569601 * np.pi * r_g**4 * (400 + 9 * alpha**2)**14)



def trans_5p_3p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5p 3p transition.
    """
    return (4613203125 * G_N * alpha**12 * (3 * (551662207031250 + 25197486328125 * alpha**2 + 461924015625 * alpha**4 + 1182886875 * alpha**6 - 419475 * alpha**8 + 82048 * alpha**10) + 700 * alpha**2 * (21015703125 + 763171875 * alpha**2 - 2156625 * alpha**4 - 8415 * alpha**6 + 64 * alpha**8) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(6576668672 * np.pi * r_g**4 * (225 + alpha**2)**16)



def trans_5d_3d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5d 3d transition.
    """
    return (389239013671875 * G_N * alpha**12 * (7 * alpha**2 * (-385287890625 - 19508343750 * alpha**2 - 179330625 * alpha**4 - 735300 * alpha**6 + 1232 * alpha**8) * np.cos(2 * theta) - 3 * (118405546875000 + 5415957421875 * alpha**2 + 95078812500 * alpha**4 + 187509375 * alpha**6 - 1610850 * alpha**8 - 8624 * alpha**10 + 57750 * alpha**4 * (-10125 + 1530 * alpha**2 + 7 * alpha**4) * np.cos( 4 * theta)))**2 * np.sin(theta)**4)/(22281753460736 * np.pi * r_g**4 * (225 + alpha**2)**16)



def trans_5p_4p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5p 4p transition.
    """
    return (429496729600000000 * G_N * alpha**12 * (6345310863360000000000000 + 47699145523200000000000 * alpha**2 + 56800051200000000 * alpha**4 + 68938866688000000 * alpha**6 + 213945528320000 * alpha**8 - 111449131200 * alpha**10 + 51265467 * alpha**12 + 1575 * alpha**2 * (11330912256000000000 - 29880483840000000 * alpha**2 + 24941363200000 * alpha**4 + 9456640000 * alpha**6 - 9357120 * alpha**8 + 2187 * alpha**10) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(817181903283661881 * np.pi * r_g**4 * (1600 + alpha**2)**18)



def trans_5d_4d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5d 4d transition.
    """
    return (4294967296000000000000 * G_N * alpha**12 * (7161992183808000000000000 + 57387012587520000000000 * alpha**2 - 26470839091200000000 * alpha**4 + 90469730304000000 * alpha**6 + 56792316160000 * alpha**8 + 26445852000 * alpha**10 - 31827411 * alpha**12 - 7 * alpha**2 * (-3312199925760000000000 - 2401822310400000000 * alpha**2 - 2616115200000000 * alpha**4 + 6704112640000 * alpha**6 - 10794124800 * alpha**8 + 1515591 * alpha**10) * np.cos( 2 * theta) + 8316000 * alpha**4 * (1719500800000 - 986624000 * alpha**2 - 381120 * alpha**4 + 567 * alpha**6) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(6229377648731354518863 * np.pi * r_g**4 * (1600 + alpha**2)**18)



def trans_5f_4f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 5f 4f transition.
    """
    return (17592186044416000000000000000000 * G_N * alpha**12 * (-215922769920000000000 - 1791177523200000000 * alpha**2 + 2374004736000000 * alpha**4 + 5062164480000 * alpha**6 + 1941977600 * alpha**8 + 382239 * alpha**10 + 96 * alpha**2 * (-7284326400000000 - 27863040000000 * alpha**2 - 5638920000 * alpha**4 + 15154075 * alpha**6 + 11583 * alpha**8) * np.cos(2 * theta) + 165 * alpha**4 * (-3452928000000 - 3588096000 * alpha**2 - 2076160 * alpha**4 + 1053 * alpha**6) * np.cos(4 * theta) - 103783680000 * alpha**6 * np.cos(6 * theta) - 64864800 * alpha**8 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(27623566774695015227961 * np.pi * r_g**4 * (1600 + alpha**2)**18)



def trans_6p_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6p 2p transition.
    """
    return 1/(1372000 * np.pi * r_g**4 * (36 + alpha**2)**16) * 59049 * G_N * alpha**12 * (7407106560 + 1895866560 * alpha**2 + 215539056 * alpha**4 + 13496868 * alpha**6 + 452223 * alpha**8 + 6181 * alpha**10 + 70 * alpha**2 * (2939328 + 711504 * alpha**2 + 65124 * alpha**4 + 2583 * alpha**6 + 35 * alpha**8) * np.cos( 2 * theta))**2 * np.sin(theta)**4



def trans_6p_3p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6p 3p transition.
    """
    return (536870912 * G_N * alpha**12 * (1132674982871040 + 85640887664640 * alpha**2 + 2864681385984 * alpha**4 + 44326969344 * alpha**6 + 125932544 * alpha**8 + 23664 * alpha**10 + 17689 * alpha**12 + 105 * alpha**2 * (112368549888 + 7978549248 * alpha**2 + 143843328 * alpha**4 - 573952 * alpha**6 - 3472 * alpha**8 + 35 * alpha**10) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(31255875 * np.pi * r_g**4 * (144 + alpha**2)**18)



def trans_6d_3d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6d 3d transition.
    """
    return (17179869184 * G_N * alpha**12 * (610257541791744 + 45910578954240 * alpha**2 + 1497299779584 * alpha**4 + 22010517504 * alpha**6 + 58196864 * alpha**8 - 632760 * alpha**10 - 4851 * alpha**12 - 7 * alpha**2 * (-706316599296 - 63302860800 * alpha**2 - 1888911360 * alpha**4 - 21260800 * alpha**6 - 106560 * alpha**8 + 231 * alpha**10) * np.cos(2 * theta) + 27720 * alpha**4 * (-331776 + 11520 * alpha**2 + 1104 * alpha**4 + 7 * alpha**6) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(272301183 * np.pi * r_g**4 * (144 + alpha**2)**18)



def trans_6p_4p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6p 4p transition.
    """
    return (253613523861504 * G_N * alpha**12 * (15389101770506443759288320 + 345982632903232003768320 * alpha**2 + 3066211089543302479872 * alpha**4 - 2772301274803077120 * alpha**6 + 16344848345333760 * alpha**8 + 162698061680640 * alpha**10 - 208181952000 * alpha**12 + 190334375 * alpha**14 + 175 * alpha**2 * (381674151054227275776 + 6885987593125625856 * alpha**2 - 44608571446394880 * alpha**4 + 70299140751360 * alpha**6 + 125625323520 * alpha**8 - 220752000 * alpha**10 + 109375 * alpha**12) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(817775726318359375 * np.pi * r_g**4 * (576 + alpha**2)**20)



def trans_6d_4d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6d 4d transition.
    """
    return (876488338465357824 * G_N * alpha**12 * (7 * alpha**2 * (-74452814972974006272 - 2053614375008206848 * alpha**2 - 3792644949934080 * alpha**4 - 6793710796800 * alpha**6 + 52072243200 * alpha**8 - 161035200 * alpha**10 + 48125 * alpha**12) * np.cos(2 * theta) - 3 * (47512679851530116923392 + 1063586136196657446912 * alpha**2 + 9122031107221487616 * alpha**4 - 6009194859724800 * alpha**6 + 52626256035840 * alpha**8 + 87779966976 * alpha**10 + 196346400 * alpha**12 - 336875 * alpha**14 + 36960 * alpha**4 * (-1430979084288 + 202569154560 * alpha**2 - 182476800 * alpha**4 - 430848 * alpha**6 + 875 * alpha**8) * np.cos( 4 * theta)))**2 * np.sin(theta)**4)/(158321380615234375 * np.pi * r_g**4 * (576 + alpha**2)**20)



def trans_6f_4f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6f 4f transition.
    """
    return (872393384948047672246272 * G_N * alpha**12 * (-83558554933697445888 - 1859494356952547328 * alpha**2 - 14692375943184384 * alpha**4 + 38833717248000 * alpha**6 + 232029388800 * alpha**8 + 220174016 * alpha**10 + 117975 * alpha**12 + 96 * alpha**2 * (-2335357865558016 - 94493541924864 * alpha**2 - 684754145280 * alpha**4 - 685948032 * alpha**6 + 1168735 * alpha**8 + 3575 * alpha**10) * np.cos(2 * theta) + 33 * alpha**4 * (15471696936960 - 444951429120 * alpha**2 - 1236566016 * alpha**4 - 1853120 * alpha**6 + 1625 * alpha**8) * np.cos(4 * theta) + 797058662400 * alpha**6 * np.cos(6 * theta) - 6088642560 * alpha**8 * np.cos(6 * theta) - 12972960 * alpha**10 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(78006744384765625 * np.pi * r_g**4 * (576 + alpha**2)**20)



def trans_6p_5p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6p 5p transition.
    """
    return (12383472844800000000 * G_N * alpha**12 * (48664433938937551257600000000000000000 + 193057365172481187840000000000000000 * alpha**2 - 82975737604074467328000000000000 * alpha**4 + 214721997260664453120000000000 * alpha**6 + 171381784885886592000000000 * alpha**8 - 136226333033991360000000 * alpha**10 + 61229396255781600000 * alpha**12 - 8190916158466800 * alpha**14 + 623903038297 * alpha**16 + 1925 * alpha**2 * (38622566618204405760000000000000 - 94477565034668851200000000000 * alpha**2 + 87071745122334720000000000 * alpha**4 - 21948899905612800000000 * alpha**6 - 2983970093760000000 * alpha**8 + 2204953081056000 * alpha**10 - 275460459120 * alpha**12 + 12400927 * alpha**14) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(27921143039686038061869103 * np.pi * r_g**4 * (3600 + alpha**2)**22)



def trans_6d_5d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6d 5d transition.
    """
    return (6687075336192000000000000 * G_N * alpha**12 * (370492301747929743360000000000000000 + 1583452828605461299200000000000000 * alpha**2 - 990703395464656896000000000000 * alpha**4 + 2401936391635614720000000000 * alpha**6 - 318800288175206400000000 * alpha**8 + 211521440422080000000 * alpha**10 - 72753371071920000 * alpha**12 + 29237722687800 * alpha**14 - 2864614137 * alpha**16 - alpha**2 * (-680666793187442688000000000000000 + 187791851639107584000000000000 * alpha**2 - 304659790800629760000000000 * alpha**4 + 364150397354112000000000 * alpha**6 - 558721772835840000000 * alpha**8 + 193731403773600000 * alpha**10 - 28740716373600 * alpha**12 + 954871379 * alpha**14) * np.cos( 2 * theta) + 7623000 * alpha**4 * (36620613900288000000000 - 38819587322880000000 * alpha**2 + 11972806732800000 * alpha**4 + 2189529792000 * alpha**6 - 996739920 * alpha**8 + 102487 * alpha**10) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(482636901114572943640880209 * np.pi * r_g**4 * (3600 + alpha**2)**22)



def trans_6f_5f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6f 5f transition.
    """
    return (103997395628457984000000000000000000 * G_N * alpha**12 * (-5719580917707571200000000000000 - 25480125370073088000000000000 * alpha**2 + 27529443515566080000000000 * alpha**4 - 17950998680524800000000 * alpha**6 - 6611812619328000000 * alpha**8 + 4784318845920000 * alpha**10 + 652988019200 * alpha**12 + 69090879 * alpha**14 + 24 * alpha**2 * (-455921255522304000000000000 - 657861782814720000000000 * alpha**2 + 493270411867200000000 * alpha**4 - 172227178260000000 * alpha**6 - 72128686770000 * alpha**8 - 23342639375 * alpha**10 + 8374652 * alpha**12) * np.cos( 2 * theta) + 165 * alpha**4 * (-37787181115392000000000 + 7066801428480000000 * alpha**2 - 2975888678400000 * alpha**4 + 3957616800000 * alpha**6 - 2447800960 * alpha**8 + 190333 * alpha**10) * np.cos( 4 * theta) - 797410489075200000000 * alpha**6 * np.cos(6 * theta) + 201981203424000000 * alpha**8 * np.cos(6 * theta) + 39932933040000 * alpha**10 * np.cos(6 * theta) - 21583762200 * alpha**12 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(1664604822211486275006301129 * np.pi * r_g**4 * (3600 + alpha**2)**22)



def trans_6g_5g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 6g 5g transition.
    """
    return (3411634563591564165120000000000000000000000000 * G_N * alpha**12 * (-8579703408427008000000000000 - 39048650128097280000000000 * alpha**2 + 69358713451315200000000 * alpha**4 + 42885582917760000000 * alpha**6 + 6756330113280000 * alpha**8 + 601035033600 * alpha**10 + 190021546 * alpha**12 + 11 * alpha**2 * (-1416617695641600000000000 - 4892770676736000000000 * alpha**2 + 1266509732544000000 * alpha**4 + 1810374446880000 * alpha**6 + 324444454800 * alpha**8 + 20831239 * alpha**10) * np.cos(2 * theta) + 286 * alpha**4 * (-32695629004800000000 - 30555527616000000 * alpha**2 - 3069925920000 * alpha**4 + 2346774000 * alpha**6 + 898909 * alpha**8) * np.cos( 4 * theta) - 1748967764544000000 * alpha**6 * np.cos(6 * theta) - 671259869280000 * alpha**8 * np.cos(6 * theta) - 152832279600 * alpha**10 * np.cos(6 * theta) + 39122083 * alpha**12 * np.cos(6 * theta) - 92185853760000 * alpha**8 * np.cos(8 * theta) - 25607181600 * alpha**10 * np.cos( 8 * theta))**2 * np.sin(theta)**4)/(173666556496502151585132390487441 * np.pi * r_g**4 * (3600 + alpha**2)**22)



def trans_7p_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7p 2p transition.
    """
    return (16925529525452800000000 * G_N * alpha**12 * (48765835668734607360 + 15723605360026583040 * alpha**2 + 2286215738735394816 * alpha**4 + 191812852308582400 * alpha**6 + 9692720764160000 * alpha**8 + 274495898250000 * alpha**10 + 3322360546875 * alpha**12 + 1575 * alpha**2 * (888590300086272 + 276389147639808 * alpha**2 + 35029983027200 * alpha**4 + 2233945280000 * alpha**6 + 70449750000 * alpha**8 + 854296875 * alpha**10) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(16677181699666569 * np.pi * r_g**4 * (784 + 25 * alpha**2)**18)



def trans_7p_3p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7p 3p transition.
    """
    return (9760113212387328 * G_N * alpha**12 * (3 * (6736540393202999694210 + 711418857763758984315 * alpha**2 + 34159102361159767551 * alpha**4 + 898249971872932560 * alpha**6 + 11390193713918880 * alpha**8 + 32224439278080 * alpha**10 + 131238240000 * alpha**12 + 6707200000 * alpha**14) + 350 * alpha**2 * (654668648513411049 + 67975280666448561 * alpha**2 + 2462592340338480 * alpha**4 + 27419673929760 * alpha**6 - 161203633920 * alpha**8 - 889056000 * alpha**10 + 12800000 * alpha**12) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(2384185791015625 * np.pi * r_g**4 * (441 + 4 * alpha**2)**20)



def trans_7d_3d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7d 3d transition.
    """
    return (474539144414574984192 * G_N * alpha**12 * (alpha**2 * (342921673030834359 + 42870471234162894 * alpha**2 + 2074962064544460 * alpha**4 + 48182589957600 * alpha**6 + 552838728960 * alpha**8 + 2724019200 * alpha**10 - 7040000 * alpha**12) * np.cos(2 * theta) + 3 * (13593569241042512568 + 1426913778042160947 * alpha**2 + 67264135264743039 * alpha**4 + 1711625900184900 * alpha**6 + 21044431164960 * alpha**8 + 50576674176 * alpha**10 - 868051200 * alpha**12 - 7040000 * alpha**14 + 8085 * alpha**4 * (-37822859361 - 490092120 * alpha**2 + 133358400 * alpha**4 + 4749696 * alpha**6 + 32000 * alpha**8) * np.cos( 4 * theta)))**2 * np.sin(theta)**4)/(461578369140625 * np.pi * r_g**4 * (441 + 4 * alpha**2)**20)



def trans_7p_4p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7p 4p transition.
    """
    return (582213431284181608955904 * G_N * alpha**12 * (72783925528663540338967474444697600 + 2809185892577184683228330794680320 * alpha**2 + 48377791529439955563156571422720 * alpha**4 + 343532022708429467209244344320 * alpha**6 - 585885062245466245756354560 * alpha**8 + 4809313411704302929182720 * alpha**10 + 55938517452998306058240 * alpha**12 - 107406579004695077184 * alpha**14 + 136231236138175299 * alpha**16 + 1155 * alpha**2 * (331559427517599946879407226880 + 12412685906967352541655859200 * alpha**2 + 81012108626534336141721600 * alpha**4 - 1096746339908566269296640 * alpha**6 + 2478623675392245104640 * alpha**8 + 8515285901737832448 * alpha**10 - 20244427195785408 * alpha**12 + 14122202241015 * alpha**14) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(10175343673354970139165125 * np.pi * r_g**4 * (3136 + 9 * alpha**2)**22)



def trans_7d_4d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7d 4d transition.
    """
    return (7455437058737706896550002688 * G_N * alpha**12 * (764492501973105535943818560405504 + 29180735756443588069577640116224 * alpha**2 + 490266606355524446037743763456 * alpha**4 + 3399381522037061466388955136 * alpha**6 - 664463084556893647011840 * alpha**8 + 35825830253785765380096 * alpha**10 + 92981026594390910976 * alpha**12 + 387315160744004064 * alpha**14 - 838858813116291 * alpha**16 + alpha**2 * (2998157821506007193363702349824 + 151559362388196945115861745664 * alpha**2 + 2178227083596396049446469632 * alpha**4 + 6838840581473736044052480 * alpha**6 + 8663916298381956218880 * alpha**8 - 148908897055847706624 * alpha**10 + 668864348134367616 * alpha**12 - 279619604372097 * alpha**14) * np.cos(2 * theta) + 76839840 * alpha**4 * (-43372684940744327168 + 1670542855782793216 * alpha**2 + 60247580204335104 * alpha**4 - 64453345173504 * alpha**6 - 337520094912 * alpha**8 + 864536409 * alpha**10) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(9849732675807611094711841 * np.pi * r_g**4 * (3136 + 9 * alpha**2)**22)



def trans_7f_4f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7f 4f transition.
    """
    return (82118921844296672746088625063340277760 * G_N * alpha**12 * (-75072535394385816769924169728 - 2837853262821170717415964672 * alpha**2 - 45621822227391945054879744 * alpha**4 - 261812610140811219173376 * alpha**6 + 1359807805730912993280 * alpha**8 + 10479573681755738112 * alpha**10 + 14580480432228480 * alpha**12 + 12239241942213 * alpha**14 + 96 * alpha**2 * (-1836225989651351834984448 - 138544032962067581370368 * alpha**2 - 3046324915646462951424 * alpha**4 - 23631287307085172736 * alpha**6 - 39129141823942464 * alpha**8 + 60114804066009 * alpha**10 + 370886119461 * alpha**12) * np.cos(2 * theta) + 33 * alpha**4 * (30193151523050877878272 + 76688897258645618688 * alpha**2 - 16107466284663570432 * alpha**4 - 68580580965765120 * alpha**6 - 147230604436608 * alpha**8 + 168584599755 * alpha**10) * np.cos(4 * theta) + 1139881460811159306240 * alpha**6 * np.cos(6 * theta) + 34711606743227301888 * alpha**8 * np.cos(6 * theta) - 296623386074695680 * alpha**10 * np.cos(6 * theta) - 1110230740146528 * alpha**12 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(1664604822211486275006301129 * np.pi * r_g**4 * (3136 + 9 * alpha**2)**22)



def trans_7p_5p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7p 5p transition.
    """
    return (3152625546875 * G_N * alpha**12 * (31121457188123367982978820800781250 + 397875779555408302901458740234375 * alpha**2 + 1936823838977197989569091796875 * alpha**4 - 4183359503630512471191406250 * alpha**6 + 13733555983464240000000000 * alpha**8 + 25866557228422435546875 * alpha**10 - 55729956176951328125 * alpha**12 + 55695182290965000 * alpha**14 - 18359452969200 * alpha**16 + 3257708544 * alpha**18 + 1050 * alpha**2 * (72586489068509313079833984375 + 750330755517294569091796875 * alpha**2 - 5398925958899021582031250 * alpha**4 + 11604055643446464843750 * alpha**6 - 5302766032798828125 * alpha**8 - 5147994761940625 * alpha**10 + 5900859153000 * alpha**12 - 1723181040 * alpha**14 + 186624 * alpha**16) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(32905425960566784 * np.pi * r_g**4 * (1225 + alpha**2)**24)



def trans_7d_5d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7d 5d transition.
    """
    return (165581804894775390625 * G_N * alpha**12 * (213895408811052237030029296875000 + 2709579309945939722137451171875 * alpha**2 + 13018069294915919976806640625 * alpha**4 - 20547881733386058105468750 * alpha**6 + 102717998975728144531250 * alpha**8 - 30287470369206640625 * alpha**10 + 75717368651203125 * alpha**12 - 80568569602500 * alpha**14 + 63518547600 * alpha**16 - 13856832 * alpha**18 + alpha**2 * (433373635849113556671142578125 + 7666864531139293542480468750 * alpha**2 - 4811520499244774902343750 * alpha**4 + 13592431550831875000000 * alpha**6 - 60843770219982421875 * alpha**8 + 202425900371406250 * alpha**10 - 166243025077500 * alpha**12 + 57995935200 * alpha**14 - 4618944 * alpha**16) * np.cos(2 * theta) + 202125 * alpha**4 * (-369686723581396484375 + 46241135208789062500 * alpha**2 - 106022797141406250 * alpha**4 + 55785584312500 * alpha**6 + 61828965625 * alpha**8 - 50757840 * alpha**10 + 11664 * alpha**12) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(47778678494742970368 * np.pi * r_g**4 * (1225 + alpha**2)**24)



def trans_7f_5f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7f 5f transition.
    """
    return (31059524496277790069580078125 * G_N * alpha**12 * (-3376274267072281281738281250000 - 42410881642211057739257812500 * alpha**2 - 190413820941101671142578125 * alpha**4 + 494220350778812109375000 * alpha**6 - 590184369590982031250 * alpha**8 - 1080512541398062500 * alpha**10 + 1302470936694375 * alpha**12 + 440885957400 * alpha**14 + 146779776 * alpha**16 + 12 * alpha**2 * (-421848624785296893310546875 - 11280422661040960205078125 * alpha**2 - 30494982392593017578125 * alpha**4 + 44563625407172265625 * alpha**6 - 29671842518500000 * alpha**8 - 30280535635000 * alpha**10 - 59591844900 * alpha**12 + 35582976 * alpha**14) * np.cos( 2 * theta) + 165 * alpha**4 * (45904007225543017578125 - 1177992407142250000000 * alpha**2 + 400247431179531250 * alpha**4 - 404612558350000 * alpha**6 + 1749828691625 * alpha**8 - 2331320040 * alpha**10 + 404352 * alpha**12) * np.cos( 4 * theta) + 8790420099281132812500 * alpha**6 * np.cos(6 * theta) - 69312481320895312500 * alpha**8 * np.cos(6 * theta) + 25068523917937500 * alpha**10 * np.cos(6 * theta) + 36393499642500 * alpha**12 * np.cos(6 * theta) - 28605376800 * alpha**14 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(2325483839696129853751296 * np.pi * r_g**4 * (1225 + alpha**2)**24)



def trans_7h_6h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7h 6h transition.
    """
    return (2054517317254333540458494019743429315435915550740648961245184 * G_N * alpha**12 * (-26911239036805619247013208040407040 - 74372046657838658633327318138880 * alpha**2 + 136200616299803735206983106560 * alpha**4 + 26118532433286248752742400 * alpha**6 + 1309074227324314214400 * alpha**8 + 493689497353153920 * alpha**10 + 61849192605480 * alpha**12 + 1550543735 * alpha**14 + 52 * alpha**2 * (-579427187965409667853339066368 - 1845309126351816925730832384 * alpha**2 + 722266264440964643291136 * alpha**4 + 298403102101394239488 * alpha**6 + 24406069659094368 * alpha**8 + 477296751078 * alpha**10 + 76585561 * alpha**12) * np.cos(2 * theta) + 1820 * alpha**4 * (-7269759579302067362070528 - 7091296524122829225984 * alpha**2 + 374614384570417152 * alpha**4 + 493148520273024 * alpha**6 + 46930672152 * alpha**8 + 1255501 * alpha**10) * np.cos(4 * theta) - 2155023875462329238814720 * alpha**6 * np.cos(6 * theta) - 731722794291857571840 * alpha**8 * np.cos(6 * theta) - 36956030721180480 * alpha**10 * np.cos(6 * theta) + 10555889804460 * alpha**12 * np.cos(6 * theta) + 2285011820 * alpha**14 * np.cos(6 * theta) - 130484434257609449472 * alpha**8 * np.cos(8 * theta) - 23283946084404096 * alpha**10 * np.cos(8 * theta) - 2484364316616 * alpha**12 * np.cos(8 * theta) + 342751773 * alpha**14 * np.cos(8 * theta) - 2578446710199360 * alpha**10 * np.cos(10 * theta) - 365426121060 * alpha**12 * np.cos( 10 * theta))**2 * np.sin(theta)**4)/(855607496999182556278970468524643316361 * np.pi * r_g**4 * (7056 + alpha**2)**26)



def trans_8i_7i(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8i 7i transition.
    """
    return (1655796070000806022650404428924209290585059099412269658845972066401620328448 * G_N * alpha**12 * (-1049303679634552282513864135533464754585600 - 1891962713926908353778608756530505318400 * alpha**2 + 3330671415718122013979940068995891200 * alpha**4 + 76687273953109312808316567552000 * alpha**6 + 2394959565019148609126400000 * alpha**8 + 5763832700958281784360960 * alpha**10 + 355327029831403929600 * alpha**12 + 1851148458036864 * alpha**14 + 156705468750 * alpha**16 + 450 * alpha**2 * (-1722202730492634392256210831692267520 - 5099710396436865275639520653475840 * alpha**2 + 2070062769991297542279030374400 * alpha**4 + 319565881073409808568156160 * alpha**6 + 8103823671469686128640 * alpha**8 + 390344056966348800 * alpha**10 + 43635082938496 * alpha**12 + 515386875 * alpha**14) * np.cos( 2 * theta) + 30600 * alpha**4 * (-8225963710663685069632595558400 - 8676078265322256247396761600 * alpha**2 + 1041367828281010264473600 * alpha**4 + 313969544804821893120 * alpha**6 + 14967572709734400 * alpha**8 + 96459408248 * alpha**10 + 8685375 * alpha**12) * np.cos(4 * theta) - 34575409490174011104622018560000 * alpha**6 * np.cos(6 * theta) - 12413743757282498146467840000 * alpha**8 * np.cos(6 * theta) + 86491349881294159872000 * alpha**10 * np.cos(6 * theta) + 209185405801021440000 * alpha**12 * np.cos(6 * theta) + 11598204229852800 * alpha**14 * np.cos(6 * theta) + 148556784375 * alpha**16 * np.cos(6 * theta) - 1971861332257849592512512000 * alpha**8 * np.cos(8 * theta) - 307241012481188954112000 * alpha**10 * np.cos(8 * theta) - 8171607181946880000 * alpha**12 * np.cos(8 * theta) + 1048343660067200 * alpha**14 * np.cos(8 * theta) + 139154456250 * alpha**16 * np.cos(8 * theta) - 48330367488281511198720 * alpha**10 * np.cos(10 * theta) - 4596321889586380800 * alpha**12 * np.cos(10 * theta) - 259490057588608 * alpha**14 * np.cos(10 * theta) + 20685121875 * alpha**16 * np.cos(10 * theta) - 423129716545536000 * alpha**12 * np.cos(12 * theta) - 33731641944000 * alpha**14 * np.cos( 12 * theta))**2 * np.sin(theta)**4)/(59527962117316694764583953656256198883056640625 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_9l_8l(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 9l 8l transition.
    """
    return (80630095298842303429095905162840439812164034993325871766922748866571777319936650504304590848 * G_N * alpha**12 * (272 * alpha**2 * (-426408178309884282694390525056710333882695680 - 1179633100740962894563268093748704056442880 * alpha**2 + 449226001100280974566956280953522094080 * alpha**4 + 22529659090090443298732151006035968 * alpha**6 - 247988244614590825378363736064 * alpha**8 + 48890273341543820147294208 * alpha**10 + 2794715535865742346240 * alpha**12 + 5489255572409160 * alpha**14 + 269651341625 * alpha**16) * np.cos(2 * theta) + 3 * (-75405042780068769292729633336159218247801440829440 - 93590262344292225532368803558981729519916810240 * alpha**2 + 154615184626186572798871841968967840169984000 * alpha**4 - 7897290923304621601224453499111751024640 * alpha**6 + 344880326147933093874852715875532800 * alpha**8 + 127705922637348764238333062676480 * alpha**10 + 2871807017489961510701629440 * alpha**12 - 2358320717223552614400 * alpha**14 + 1663621228621900800 * alpha**16 + 10133213574750 * alpha**18 + 1615 * alpha**4 * (-5822009164015650550954972756828634480640 - 6670132502121819989932312012951388160 * alpha**2 + 947722542844296990215226569785344 * alpha**4 + 111189382825001011752141324288 * alpha**6 + 1935030641553189554356224 * alpha**8 + 10852598726891274240 * alpha**10 + 1760067649774080 * alpha**12 + 10905566725 * alpha**14) * np.cos(4 * theta) + 90440 * alpha**6 * (-12028802211687012130367797513420800 - 4814959347003467865491238813696 * alpha**2 + 203744298899173202416631808 * alpha**4 + 49405753216607100862464 * alpha**6 + 1476604041406156800 * alpha**8 + 3868489046160 * alpha**10 + 209147855 * alpha**12) * np.cos( 6 * theta) - 55494044896796630756406039165272064 * alpha**8 * np.cos( 8 * theta) - 9044847570842148420248307499008 * alpha**10 * np.cos( 8 * theta) - 29105518831818076726493184 * alpha**12 * np.cos(8 * theta) + 44486913001120951173120 * alpha**14 * np.cos(8 * theta) + 1538903752947348480 * alpha**16 * np.cos(8 * theta) + 10403432603410 * alpha**18 * np.cos(8 * theta) - 1353067683365876139561729392640 * alpha**10 * np.cos( 10 * theta) - 111750830558467022955479040 * alpha**12 * np.cos(10 * theta) - 1653002446709093990400 * alpha**14 * np.cos(10 * theta) + 108096589932312960 * alpha**16 * np.cos(10 * theta) + 9341857847960 * alpha**18 * np.cos(10 * theta) - 15562320652089508296130560 * alpha**12 * np.cos(12 * theta) - 864868991715246735360 * alpha**14 * np.cos(12 * theta) - 28217287965335040 * alpha**16 * np.cos(12 * theta) + 1380047182085 * alpha**18 * np.cos(12 * theta) - 67871980316838297600 * alpha**14 * np.cos(14 * theta) - 3273147198921600 * alpha**16 * np.cos( 14 * theta)))**2 * np.sin(theta)**4)/(763103482836293187320433699957796773008680132070767507225 * np.pi * r_g**4 * (20736 + alpha**2)**34)




diff_power_ann_dict = {
    "2p": ann_2p,
    "3p": ann_3p,
    "3d": ann_3d,
    "4p": ann_4p,
    "4d": ann_4d,
    "4f": ann_4f,
    "5p": ann_5p,
    "5d": ann_5d,
    "5f": ann_5f,
    "5g": ann_5g,
    "6p": ann_6p,
    "6d": ann_6d,
    "6f": ann_6f,
    "6g": ann_6g,
    "6h": ann_6h
}


diff_power_trans_dict = {
    "3p 2p": trans_3p_2p,
    "4p 2p": trans_4p_2p,
    "4p 3p": trans_4p_3p,
    "4d 3d": trans_4d_3d,
    "5p 2p": trans_5p_2p,
    "5p 3p": trans_5p_3p,
    "5d 3d": trans_5d_3d,
    "5p 4p": trans_5p_4p,
    "5d 4d": trans_5d_4d,
    "5f 4f": trans_5f_4f,
    "6p 2p": trans_6p_2p,
    "6p 3p": trans_6p_3p,
    "6d 3d": trans_6d_3d,
    "6p 4p": trans_6p_4p,
    "6d 4d": trans_6d_4d,
    "6f 4f": trans_6f_4f,
    "6p 5p": trans_6p_5p,
    "6d 5d": trans_6d_5d,
    "6f 5f": trans_6f_5f,
    "6g 5g": trans_6g_5g,
    "7p 2p": trans_7p_2p,
    "7p 3p": trans_7p_3p,
    "7d 3d": trans_7d_3d,
    "7p 4p": trans_7p_4p,
    "7d 4d": trans_7d_4d,
    "7f 4f": trans_7f_4f,
    "7p 5p": trans_7p_5p,
    "7d 5d": trans_7d_5d,
    "7f 5f": trans_7f_5f,
    "7h 6h": trans_7h_6h,
    "8i 7i": trans_8i_7i,
    "9l 8l": trans_9l_8l
}
