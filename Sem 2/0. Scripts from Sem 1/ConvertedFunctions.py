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



def ann_7p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7p annihilation.
    """
    return 1/(25 * np.pi * r_g**4 * (9604 + alpha**4)**28) * 9834496 * G_N * alpha**20 * (-98 + alpha**2)**2 * (8335219048956705762975832185266713110532587520 - 3365445166016019093522676355808378249915924480 * alpha**2 + 638536505762181935298942887008549462945562624 * alpha**4 - 73334643339951250377678154670309109905489920 * alpha**6 + 5658661971010178209110772262331315007258624 * alpha**8 - 310830923947126820004126131382834628984832 * alpha**10 + 12610174246511678622101307396373031813120 * alpha**12 - 387409491999896042787956683909175050240 * alpha**14 + 9172064030590763736395469945204834304 * alpha**16 - 169390642047065610767806531790938112 * alpha**18 + 2459948357223670207678401953054720 * alpha**20 - 28220906619441947612685306081280 * alpha**22 + 256137896420623720083132231680 * alpha**24 - 1836475383615427605044460032 * alpha**26 + 10354058722777360234126336 * alpha**28 - 45536710190369559831040 * alpha**30 + 154333544577835978880 * alpha**32 - 396105899805155792 * alpha**34 + 750842231515936 * alpha**36 - 1013192515720 * alpha**38 + 918578696 * alpha**40 - 504105 * alpha**42 + 130 * alpha**44)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta))



def ann_7d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7d annihilation.
    """
    return 1/(1089 * np.pi * r_g**4 * (9604 + alpha**4)**28) * 30118144 * G_N * alpha**20 * (-98 + alpha**2)**6 * (1982795676112630643965512894486442983555072 - 1073466905487958351994050977773029649547264 * alpha**2 + 230978574686702272238037778317212608102400 * alpha**4 - 27675979672617734565092682100124977987584 * alpha**6 + 2109435197997537712801584095907731537920 * alpha**8 - 109783577018614156700962395149541310464 * alpha**10 + 4073207069196803174195179556142448640 * alpha**12 - 110797154152207118325218204944564224 * alpha**14 + 2250990543441442999316017612431360 * alpha**16 - 34566049212976243082534123372544 * alpha**18 + 403901330026745597390299017216 * alpha**20 - 3599130488648088617506676736 * alpha**22 + 24404469289588693080168960 * alpha**24 - 125075472280003029743616 * alpha**26 + 478771051524543357440 * alpha**28 - 1343620496156391936 * alpha**30 + 2688148645485520 * alpha**32 - 3672298229376 * alpha**34 + 3191203400 * alpha**36 - 1544256 * alpha**38 + 297 * alpha**40)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**4



def ann_7f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7f annihilation.
    """
    return 1/(1521 * np.pi * r_g**4 * (9604 + alpha**4)**28) * 9445049958400 * G_N * alpha**24 * (-98 + alpha**2)**10 * (42342166654661006333614491735490560 - 14491833647556792879382814400626688 * alpha**2 + 2057499023372092946782191599738880 * alpha**4 - 163505733743656481921702086037504 * alpha**6 + 8193617316726743344204997824512 * alpha**8 - 275610093432467483495473110016 * alpha**10 + 6454065332699073840795738624 * alpha**12 - 107545964671786142309282048 * alpha**14 + 1290603251441455846272000 * alpha**16 - 11198038803809469211712 * alpha**18 + 69972768061498066464 * alpha**20 - 311127689740530544 * alpha**22 + 963090437553552 * alpha**24 - 2001115841396 * alpha**26 + 2621964030 * alpha**28 - 1922907 * alpha**30 + 585 * alpha**32)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**8



def ann_7g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7g annihilation.
    """
    return (11112006825558016 * G_N * alpha**28 * (-98 + alpha**2)**14 * (460036255705907845572666347520 - 100413120671201732076250365952 * alpha**2 + 8963522724553219319006361600 * alpha**4 - 435590222442660812409774080 * alpha**6 + 12860412954309586549820160 * alpha**8 - 243623135941397060423680 * alpha**10 + 3045188257781261625600 * alpha**12 - 25366840477030097920 * alpha**14 + 139428197026115760 * alpha**16 - 491724297518720 * alpha**18 + 1053586308600 * alpha**20 - 1228937248 * alpha**22 + 586245 * alpha**24)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**12)/(12623809 * np.pi * r_g**4 * (9604 + alpha**4)**28)



def ann_7h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7h annihilation.
    """
    return 1/(529 * np.pi * r_g**4 * (9604 + alpha**4)**28) * 983528880101307055079424 * G_N * alpha**32 * (-98 + alpha**2)**18 * (587026485581432064 - 71992410292817280 * alpha**2 + 3391989907615904 * alpha**4 - 80929504726560 * alpha**6 + 1081262075824 * alpha**8 - 8426645640 * alpha**10 + 36774794 * alpha**12 - 81270 * alpha**14 + 69 * alpha**16)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**16



def ann_7i(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7i annihilation.
    """
    return 1/(5625 * np.pi * r_g**4 * (9604 + alpha**4)**28) * 161981742503351375821275136 * G_N * alpha**36 * (-98 + alpha**2)**22 * (103766418000 - 4389719488 * alpha**2 + 65556904 * alpha**4 - 457072 * alpha**6 + 1125 * alpha**8)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**20



def ann_8p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p annihilation.
    """
    return 1/(1225 * np.pi * r_g**4 * (16384 + alpha**4)**32) * 4194304 * G_N * alpha**20 * (-128 + alpha**2)**2 * (21755306698132158233255094397425789170651620578518416490496 - 8910791103336014884978256186054669642475134797131691327488 * alpha**2 + 1722163938568772850537869020215873291830286971758952054784 * alpha**4 - 202847868707941115279182454002639696574880331824097132544 * alpha**6 + 16183305459430936850583489411820891822844769267357843456 * alpha**8 - 927765196421885199749469365906146759602851599446179840 * alpha**10 + 39704242096183361247922612017041953409902068272988160 * alpha**12 - 1302474983771203231562241648146777179606957588217856 * alpha**14 + 33386226363611975781070641131737921135070515560448 * alpha**16 - 678230250747897884225049434513330902316869484544 * alpha**18 + 11033904237192376650732857832693335807320129536 * alpha**20 - 144838261397016542930276792800465570693644288 * alpha**22 + 1541851821473181695141616779868166741295104 * alpha**24 - 13350056466588181243397813301822797905920 * alpha**26 + 94107166837962749947608446036875411456 * alpha**28 - 539564570028396483250993463398760448 * alpha**30 + 2508819360899082460203512217206784 * alpha**32 - 9412335504720119097946804322304 * alpha**34 + 28279233714469474094811185152 * alpha**36 - 67336378431928666807074816 * alpha**38 + 125284497444437638512640 * alpha**40 - 178681077314921431040 * alpha**42 + 190233841451925504 * alpha**44 - 145536208338944 * alpha**46 + 75414552576 * alpha**48 - 23816448 * alpha**50 + 3549 * alpha**52)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta))



def ann_8d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8d annihilation.
    """
    return 1/(53361 * np.pi * r_g**4 * (16384 + alpha**4)**32) * 4194304 * G_N * alpha**20 * (-128 + alpha**2)**6 * (5444923731987617323166265694853137096973513292270010368 - 3022072624649918421620724645503072000082262129310171136 * alpha**2 + 675203845638898998024688323977150897786813793062354944 * alpha**4 - 84993735297773496919861326214920386263786661886296064 * alpha**6 + 6895710595547263428968270796286008142418421552775168 * alpha**8 - 387713939648614789821490362030548151529412367482880 * alpha**10 + 15802366799981566210024965395599911613657692241920 * alpha**12 - 481172691859487845524951249564593682293257666560 * alpha**14 + 11178585059593035269038675355933974098230640640 * alpha**16 - 201120769127467326713754442328518997720432640 * alpha**18 + 2831697183534734165952464716318542282096640 * alpha**20 - 31417253290092864824222727459566140784640 * alpha**22 + 275770392437548259498962699092995604480 * alpha**24 - 1917556963506644581556562955295784960 * alpha**26 + 10548894046003871284248175905341440 * alpha**28 - 45729568484481962221651529564160 * alpha**30 + 155134031448637067355215626240 * alpha**32 - 407569123324027278395965440 * alpha**34 + 816963215587273730949120 * alpha**36 - 1223409477592044011520 * alpha**38 + 1328065552379281408 * alpha**40 - 999096556978176 * alpha**42 + 484435001344 * alpha**44 - 132338304 * alpha**46 + 14553 * alpha**48)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**4



def ann_8f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8f annihilation.
    """
    return 1/(184041 * np.pi * r_g**4 * (16384 + alpha**4)**32) * 1717986918400 * G_N * alpha**24 * (-128 + alpha**2)**10 * (98659890554874384942100363620588391825051484160 - 36040900724332718024922720973258211404680265728 * alpha**2 + 5570836640277488889474708982242593799332167680 * alpha**4 - 492059698039284882251818086684264617159426048 * alpha**6 + 28055545700127442863965448437657729312489472 * alpha**8 - 1102380081665171362372497978141562760069120 * alpha**10 + 31063661879827298584710416994626363719680 * alpha**12 - 644154581065858794300877266206846877696 * alpha**14 + 9998691719827745148392882721402126336 * alpha**16 - 117450291637815537077131128491474944 * alpha**18 + 1050497469738751658328375032807424 * alpha**20 - 7168596901722139714180366729216 * alpha**22 + 37248029261185769544515321856 * alpha**24 - 146463794650539692679757824 * alpha**26 + 431094908101937992826880 * alpha**28 - 933752249567286394880 * alpha**30 + 1450437717358804992 * alpha**32 - 1552666635272192 * alpha**34 + 1072904110080 * alpha**36 - 423658752 * alpha**38 + 70785 * alpha**40)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**8



def ann_8g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8g annihilation.
    """
    return (882705526964617216 * G_N * alpha**28 * (-128 + alpha**2)**14 * (3043958071831750023087860820523635179520 - 763058161478233146941844433184072138752 * alpha**2 + 81021203582263158952784977151876136960 * alpha**4 - 4866428113386113280330634431431704576 * alpha**6 + 185493277467457232977185714607226880 * alpha**8 - 4763639743185343648868277600387072 * alpha**10 + 85332970617375284503891043942400 * alpha**12 - 1088566803084670439480612618240 * alpha**14 + 10001995083959984022072852480 * alpha**16 - 66440844914835842253455360 * alpha**18 + 317890087579843716710400 * alpha**20 - 1083126276895505645568 * alpha**22 + 2574236344469422080 * alpha**24 - 4122024947482624 * alpha**26 + 4188698050560 * alpha**28 - 2407787008 * alpha**30 + 586245 * alpha**32)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**12)/(12623809 * np.pi * r_g**4 * (16384 + alpha**4)**32)



def ann_8h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8h annihilation.
    """
    return 1/(89401 * np.pi * r_g**4 * (16384 + alpha**4)**32) * 42501298345826806923264 * G_N * alpha**32 * (-128 + alpha**2)**18 * (1578895806042933370299856453632 - 259638731542722130421942845440 * alpha**2 + 17633876637759285639604338688 * alpha**4 - 654888283771164437816279040 * alpha**6 + 14806425767966838787932160 * alpha**8 - 214886883385929545809920 * alpha**10 + 2057131609840995532800 * alpha**12 - 13115654503535738880 * alpha**14 + 55158234268303360 * alpha**16 - 148904356085760 * alpha**18 + 244719198208 * alpha**20 - 219922560 * alpha**22 + 81627 * alpha**24)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**16



def ann_8i(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8i annihilation.
    """
    return 1/(5625 * np.pi * r_g**4 * (16384 + alpha**4)**32) * 9361921547095688328924626944 * G_N * alpha**36 * (-128 + alpha**2)**22 * (567453553048682496000 - 53079706683165376512 * alpha**2 + 1916268447306612736 * alpha**4 - 35033240779620352 * alpha**6 + 358669497663488 * alpha**8 - 2138259324928 * alpha**10 + 7138656256 * alpha**12 - 12068928 * alpha**14 + 7875 * alpha**16)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**20



def ann_8k(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8k annihilation.
    """
    return 1/(808201 * np.pi * r_g**4 * (16384 + alpha**4)**32) * 58324921029150891820213688655151104 * G_N * alpha**40 * (-128 + alpha**2)**26 * (723970424832 - 23282581504 * alpha**2 + 267288576 * alpha**4 - 1421056 * alpha**6 + 2697 * alpha**8)**2 * (35 + 28 * np.cos(2 * theta) + np.cos(4 * theta)) * np.sin(theta)**24




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



def trans_7p_6p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7p 6p transition.
    """
    return (17909885825636445978624 * G_N * alpha**12 * (3 * (9695212879003258881609608676965197878135336468480 + 22515555273491558412688354158770164006177996800 * alpha**2 - 12041422367315096888348651622178529457733632 * alpha**4 + 17949759935027437130714138594970911637504 * alpha**6 + 2867785348087009543486060626406539264 * alpha**8 - 4304406028621813566671849848307712 * alpha**10 + 1921919032285635503075911925760 * alpha**12 - 326880288611482994606260224 * alpha**14 + 30519491373783542527488 * alpha**16 - 1252228899613731312 * alpha**18 + 26161299953191 * alpha**20) + 455 * alpha**2 * (58887347418630095247871772819273553681580032 - 122772060626537127797066793692972810502144 * alpha**2 + 101810685059327950297110096764454567936 * alpha**4 - 34397902049447883956657805085114368 * alpha**6 + 4172835238561908970842300088320 * alpha**8 + 285029958135279694007107584 * alpha**10 - 107728317117023661047808 * alpha**12 + 9256629908181906432 * alpha**14 - 309893417202096 * alpha**16 + 4078653605 * alpha**18) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(11466666274158577082299952001125 * np.pi * r_g**4 * (7056 + alpha**2)**26)



def trans_7d_6d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7d 6d transition.
    """
    return (111460240168179252811800772608 * G_N * alpha**12 * (alpha**2 * (-18147698101233763969336967127770580010401792 + 9770316669627950321176028127770192117760 * alpha**2 - 7255286209540751840242020249174540288 * alpha**4 + 4247542084385505342421294682996736 * alpha**6 - 4185781336178413764803165159424 * alpha**8 + 1545392366729090226795184128 * alpha**10 - 278808194869142000025600 * alpha**12 + 22788721755954266112 * alpha**14 - 854524055234304 * alpha**16 + 8973037931 * alpha**18) * np.cos( 2 * theta) + 3 * (-5246503172213713736890113519530790567159005184 - 13283120258426125893319501817358864875520000 * alpha**2 + 8062396804512525818235192777302008135680 * alpha**4 - 13826305519942604227188156595867484160 * alpha**6 + 4185802544955703657015106517073920 * alpha**8 - 1329452792744335997598328946688 * alpha**10 + 227668249901768777938206720 * alpha**12 - 51026390638105936435200 * alpha**14 + 5849129355898308480 * alpha**16 - 381264429956280 * alpha**18 + 8973037931 * alpha**20 - 840840 * alpha**4 * (2072011904184526067655285267234816 - 2259403499421194982329992347648 * alpha**2 + 907374149919306721117863936 * alpha**4 - 106954623704643565584384 * alpha**6 - 8533868438950170624 * alpha**8 + 2907477496804608 * alpha**10 - 198964151568 * alpha**12 + 4826809 * alpha**14) * np.cos( 4 * theta)))**2 * np.sin(theta)**4)/(11099732953385502615666353537089 * np.pi * r_g**4 * (7056 + alpha**2)**26)



def trans_7g_6g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7g 6g transition.
    """
    return (2193134958446163176101316494214495135762202230784 * G_N * alpha**12 * (-51316473541124797393924862515998948851712 - 140114419093612505525110020353128660992 * alpha**2 + 178574790323453625316814658855763968 * alpha**4 - 71314235632570647682422820306944 * alpha**6 + 3284516573298006504764866560 * alpha**8 + 5923790891323344345341952 * alpha**10 + 536783579006195243520 * alpha**12 - 13191877693612800 * alpha**14 + 2466911534230 * alpha**16 + 11 * alpha**2 * (-5691766058080093986133506563166437376 - 9579611553948240386979497716482048 * alpha**2 + 7494134076918206922990107492352 * alpha**4 - 1284068235672593013347647488 * alpha**6 - 486916748867164227526656 * alpha**8 + 64682382574886920704 * alpha**10 + 7492185730852368 * alpha**12 + 270436825945 * alpha**14) * np.cos(2 * theta) + 286 * alpha**4 * (-98645273079830445181418619273216 - 11326026838083068412061286400 * alpha**2 + 12549101116492321602600960 * alpha**4 - 1729651439297011507200 * alpha**6 - 381429536411911680 * alpha**8 - 78975311500176 * alpha**10 + 11669881795 * alpha**12) * np.cos( 4 * theta) - 4078156732286687257305208061952 * alpha**6 * np.cos(6 * theta) + 616598721009673013564080128 * alpha**8 * np.cos(6 * theta) - 48828194223745441849344 * alpha**10 * np.cos(6 * theta) + 38009054141913254400 * alpha**12 * np.cos(6 * theta) - 11909690260484016 * alpha**14 * np.cos(6 * theta) + 507893551165 * alpha**16 * np.cos(6 * theta) - 160105912162988227683287040 * alpha**8 * np.cos(8 * theta) + 20724792333207266304000 * alpha**10 * np.cos(8 * theta) + 2262755147767994880 * alpha**12 * np.cos(8 * theta) - 551337984156960 * alpha**14 * np.cos( 8 * theta))**2 * np.sin(theta)**4)/(2152766689047092594342740441599811444379 * np.pi * r_g**4 * (7056 + alpha**2)**26)



def trans_7h_6h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 7h 6h transition.
    """
    return (2054517317254333540458494019743429315435915550740648961245184 * G_N * alpha**12 * (-26911239036805619247013208040407040 - 74372046657838658633327318138880 * alpha**2 + 136200616299803735206983106560 * alpha**4 + 26118532433286248752742400 * alpha**6 + 1309074227324314214400 * alpha**8 + 493689497353153920 * alpha**10 + 61849192605480 * alpha**12 + 1550543735 * alpha**14 + 52 * alpha**2 * (-579427187965409667853339066368 - 1845309126351816925730832384 * alpha**2 + 722266264440964643291136 * alpha**4 + 298403102101394239488 * alpha**6 + 24406069659094368 * alpha**8 + 477296751078 * alpha**10 + 76585561 * alpha**12) * np.cos(2 * theta) + 1820 * alpha**4 * (-7269759579302067362070528 - 7091296524122829225984 * alpha**2 + 374614384570417152 * alpha**4 + 493148520273024 * alpha**6 + 46930672152 * alpha**8 + 1255501 * alpha**10) * np.cos(4 * theta) - 2155023875462329238814720 * alpha**6 * np.cos(6 * theta) - 731722794291857571840 * alpha**8 * np.cos(6 * theta) - 36956030721180480 * alpha**10 * np.cos(6 * theta) + 10555889804460 * alpha**12 * np.cos(6 * theta) + 2285011820 * alpha**14 * np.cos(6 * theta) - 130484434257609449472 * alpha**8 * np.cos(8 * theta) - 23283946084404096 * alpha**10 * np.cos(8 * theta) - 2484364316616 * alpha**12 * np.cos(8 * theta) + 342751773 * alpha**14 * np.cos(8 * theta) - 2578446710199360 * alpha**10 * np.cos(10 * theta) - 365426121060 * alpha**12 * np.cos( 10 * theta))**2 * np.sin(theta)**4)/(855607496999182556278970468524643316361 * np.pi * r_g**4 * (7056 + alpha**2)**26)



def trans_8p_2p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p 2p transition.
    """
    return (14023813415445725184 * G_N * alpha**12 * (105924663235754065920 + 41110827723474862080 * alpha**2 + 7271369461645443072 * alpha**4 + 761073122588753920 * alpha**6 + 50713401973800960 * alpha**8 + 2131973386076160 * alpha**10 + 51713976960000 * alpha**12 + 550693434375 * alpha**14 + 525 * alpha**2 * (5910974510923776 + 2236406650896384 * alpha**2 + 359467287838720 * alpha**4 + 31234814115840 * alpha**6 + 1536229048320 * alpha**8 + 40170816000 * alpha**10 + 430565625 * alpha**12) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(817775726318359375 * np.pi * r_g**4 * (256 + 9 * alpha**2)**20)



def trans_8p_3p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p 3p transition.
    """
    return (155820149060508057600000000 * G_N * alpha**12 * (3 * (169775207473918788632418629713920 + 22991233048486373569689896878080 * alpha**2 + 1433425805245949322105288916992 * alpha**4 + 51965594166311310777856819200 * alpha**6 + 1101883573701502304256000000 * alpha**8 + 11595432949545959424000000 * alpha**10 + 35085129237734400000000 * alpha**12 + 354415236037500000000 * alpha**14 + 8371231239013671875 * alpha**16) + 1925 * alpha**2 * (3158020972357120324263739392 + 427824737642853159841824768 * alpha**2 + 22706019602883729476812800 * alpha**4 + 539307133105707417600000 * alpha**6 + 3867687457652736000000 * alpha**8 - 34955415475200000000 * alpha**10 - 122215747500000000 * alpha**12 + 3027570068359375 * alpha**14) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(27921143039686038061869103 * np.pi * r_g**4 * (2304 + 25 * alpha**2)**22)



def trans_8d_3d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8d 3d transition.
    """
    return (168285760985348702208000000000000 * G_N * alpha**12 * (7 * alpha**2 * (-6616805846843490203219263488 - 1047499780291744482195406848 * alpha**2 - 68722802758619046818611200 * alpha**4 - 2388301303102046208000000 * alpha**6 - 46846432650461184000000 * alpha**8 - 505055024686080000000 * alpha**10 - 2223074799000000000 * alpha**12 + 6660654150390625 * alpha**14) * np.cos(2 * theta) - 3 * (1793543608367929579790256832512 + 241448542764230103297863712768 * alpha**2 + 14838007708786618700132253696 * alpha**4 + 525037846320773585908531200 * alpha**6 + 10841756288038962462720000 * alpha**8 + 111908174873178931200000 * alpha**10 + 164742829954560000000 * alpha**12 - 6197941587750000000 * alpha**14 - 46624579052734375 * alpha**16 + 1626240 * alpha**4 * (-30709554377341796352 - 1564654545823334400 * alpha**2 + 76120137400320000 * alpha**4 + 6936474009600000 * alpha**6 + 156968460000000 * alpha**8 + 1000849609375 * alpha**10) * np.cos( 4 * theta)))**2 * np.sin(theta)**4)/(3378458307802010605486161463 * np.pi * r_g**4 * (2304 + 25 * alpha**2)**22)



def trans_8p_4p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p 4p transition.
    """
    return (274877906944 * G_N * alpha**12 * (506757147276741627881717760 + 27936025926386850610544640 * alpha**2 + 709222257632287282692096 * alpha**4 + 9605804206412196216832 * alpha**6 + 49693193277429252096 * alpha**8 - 98543085394329600 * alpha**10 + 1751158494330880 * alpha**12 + 15599764242432 * alpha**14 - 38551666176 * alpha**16 + 60975747 * alpha**18 + 105 * alpha**2 * (28278858664996742627328 + 1584978838458262880256 * alpha**2 + 27654353511868530688 * alpha**4 + 29884726042951680 * alpha**6 - 2149845880012800 * alpha**8 + 7210780327936 * alpha**10 + 29896998912 * alpha**12 - 88833024 * alpha**14 + 76545 * alpha**16) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(4036388792207625 * np.pi * r_g**4 * (256 + alpha**2)**24)



def trans_8d_4d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8d 4d transition.
    """
    return (4398046511104 * G_N * alpha**12 * (148357865425832774033473536 + 8084912779601717120466944 * alpha**2 + 201083093259910262030336 * alpha**4 + 2654768960493431291904 * alpha**6 + 15074693094169378816 * alpha**8 + 11085237576531968 * alpha**10 + 231124261404672 * alpha**12 + 829504094208 * alpha**14 + 4026817152 * alpha**16 - 10609137 * alpha**18 - 7 * alpha**2 * (-86847271099024569008128 - 6201780946062343667712 * alpha**2 - 158837736732706734080 * alpha**4 - 1698163823262826496 * alpha**6 - 6631794577244160 * alpha**8 - 528968843264 * alpha**10 + 156071952384 * alpha**12 - 991284480 * alpha**14 + 505197 * alpha**16) * np.cos(2 * theta) + 147840 * alpha**4 * (-7036874417766400 + 54975581388800 * alpha**2 + 12305081303040 * alpha**4 + 175489679360 * alpha**6 - 245825536 * alpha**8 - 1617408 * alpha**10 + 5103 * alpha**12) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(3907224350856981 * np.pi * r_g**4 * (256 + alpha**2)**24)



def trans_8f_4f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8f 4f transition.
    """
    return (1441151880758558720 * G_N * alpha**12 * (-1392802964541365985214464 - 75179705472403277611008 * alpha**2 - 1813307117441296367616 * alpha**4 - 22119127204132552704 * alpha**6 - 77751393527529472 * alpha**8 + 946068585971712 * alpha**10 + 7055381692416 * alpha**12 + 11935224576 * alpha**14 + 12613887 * alpha**16 + 96 * alpha**2 * (-30912707842271084544 - 3331537824347324416 * alpha**2 - 119238857756508160 * alpha**4 - 1937083669151744 * alpha**6 - 13908725727232 * alpha**8 - 27715858432 * alpha**10 + 48123108 * alpha**12 + 382239 * alpha**14) * np.cos(2 * theta) + 33 * alpha**4 * (788411409766547456 + 19875871695306752 * alpha**2 - 291117178290176 * alpha**4 - 10392243798016 * alpha**6 - 53096022016 * alpha**8 - 126542592 * alpha**10 + 173745 * alpha**12) * np.cos( 4 * theta) + 24763750636584960 * alpha**6 * np.cos(6 * theta) + 1625121135525888 * alpha**8 * np.cos(6 * theta) + 16777199222784 * alpha**10 * np.cos(6 * theta) - 220814770176 * alpha**12 * np.cos(6 * theta) - 1027458432 * alpha**14 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(190588252519498977 * np.pi * r_g**4 * (256 + alpha**2)**24)



def trans_8p_5p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p 5p transition.
    """
    return (3462669979122401280000000 * G_N * alpha**12 * (2289005590221855446051848192000000000000000000000 + 51680846272371469877027799040000000000000000000 * alpha**2 + 520312893957304443612262563840000000000000000 * alpha**4 + 1806268764564317199473521459200000000000000 * alpha**6 - 7815461027971066187835506688000000000000 * alpha**8 + 42483536614637911703751229440000000000 * alpha**10 + 96805021746622087059996672000000000 * alpha**12 - 349154462951374554053738496000000 * alpha**14 + 518913184167989212114206720000 * alpha**16 - 259865981597617981755436800 * alpha**18 + 68091257647190919584079 * alpha**20 + 6825 * alpha**2 * (1021877495634756895558860800000000000000000 + 23036505885166706538577920000000000000000 * alpha**2 + 44023995410757880601640960000000000000 * alpha**4 - 1342814129286995198450073600000000000 * alpha**6 + 4806950510655232152698880000000000 * alpha**8 - 2620772858652614637649920000000 * alpha**10 - 6943554688848510320640000000 * alpha**12 + 10506155078003467124736000 * alpha**14 - 4546876265090965128960 * alpha**16 + 737405187918332661 * alpha**18) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(31464532256291135513831068291087 * np.pi * r_g**4 * (6400 + 9 * alpha**2)**26)



def trans_8d_5d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8d 5d transition.
    """
    return (76948221758275584000000000000 * G_N * alpha**12 * (82025411179656167642313523200000000000000000000 + 1821738847222040717393133568000000000000000000 * alpha**2 + 17970356314193725202764922880000000000000000 * alpha**4 + 63893600669187661183215206400000000000000 * alpha**6 - 103744835983798552642977792000000000000 * alpha**8 + 1182130827142338620298362880000000000 * alpha**10 - 413972043509163344737075200000000 * alpha**12 + 2523886163197950312972288000000 * alpha**14 - 4422551558869339230965760000 * alpha**16 + 4862810761596841928284800 * alpha**18 - 1533065385682213602219 * alpha**20 + alpha**2 * (177630550282078368888258560000000000000000000 + 5891730093019239527082885120000000000000000 * alpha**2 + 46040805288825274305714585600000000000000 * alpha**4 - 14759733645052908429901824000000000000 * alpha**6 + 64315313150101080370053120000000000 * alpha**8 - 1187435651602630878167040000000000 * alpha**10 + 6642051706193527777001472000000 * alpha**12 - 8274289096755403862999040000 * alpha**14 + 4302262141142069723673600 * alpha**16 - 511021795227404534073 * alpha**18) * np.cos(2 * theta) + 1297296000 * alpha**4 * (-104278474426647838720000000000000 + 3576284563460902092800000000000 * alpha**2 + 66824038207916605440000000000 * alpha**4 - 243504399843223142400000000 * alpha**6 + 148750772106166272000000 * alpha**8 + 435167868574752768000 * alpha**10 - 485303567117541120 * alpha**12 + 161605344711447 * alpha**14) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(543886914715889628167651323317361 * np.pi * r_g**4 * (6400 + 9 * alpha**2)**26)



def trans_8f_5f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8f 5f transition.
    """
    return (100857573223006973460480000000000000000000 * G_N * alpha**12 * (-418475630296771095166976000000000000000000 - 9176675608434455619829760000000000000000 * alpha**2 - 86934173008234077487104000000000000000 * alpha**4 - 260460355982665407528960000000000000 * alpha**6 + 1185915842264094285496320000000000 * alpha**8 - 2139845980878932646297600000000 * alpha**10 - 7367127004606525341696000000 * alpha**12 + 12328482852026840678400000 * alpha**14 + 6125070545429584320000 * alpha**16 + 3595187930839929567 * alpha**18 + 96 * alpha**2 * (-5426226750281985556480000000000000000 - 289372995232366329856000000000000000 * alpha**2 - 3646291407212025741312000000000000 * alpha**4 - 10447995195955410370560000000000 * alpha**6 + 20185524162696314880000000000 * alpha**8 - 20177075065314336768000000 * alpha**10 - 24608982149861616000000 * alpha**12 - 139816061325268742700 * alpha**14 + 108945088813331199 * alpha**16) * np.cos(2 * theta) + 165 * alpha**4 * (13101241619959345315840000000000000 - 54017172479654284492800000000000 * alpha**2 - 3801065700379475312640000000000 * alpha**4 + 1098618586951738982400000000 * alpha**6 - 1916575594926833664000000 * alpha**8 + 20335217131023335424000 * alpha**10 - 39593769058872737280 * alpha**12 + 9904098983030109 * alpha**14) * np.cos(4 * theta) + 1894184548489393864704000000000000 * alpha**6 * np.cos( 6 * theta) + 31661482739425188249600000000000 * alpha**8 * np.cos(6 * theta) - 364716669310693539840000000000 * alpha**10 * np.cos(6 * theta) + 118036520294795182080000000 * alpha**12 * np.cos(6 * theta) + 567399701513201356800000 * alpha**14 * np.cos(6 * theta) - 563162305250450505600 * alpha**16 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(20634403560343649362523751225448451 * np.pi * r_g**4 * (6400 + 9 * alpha**2)**26)



def trans_8g_5g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8g 5g transition.
    """
    return (4957351439057238759529512960000000000000000000000000 * G_N * alpha**12 * (-432674867432320946667520000000000000000 - 9365954762567428184473600000000000000 * alpha**2 - 82086990042345114697728000000000000 * alpha**4 - 115204729260040425308160000000000 * alpha**6 + 2756369824833727365120000000000 * alpha**8 + 6658454579478633381888000000 * alpha**10 + 4111489030872698634240000 * alpha**12 + 546244890112220083200 * alpha**14 + 2359832278792905774 * alpha**16 + 33 * alpha**2 * (-2363826854416089088000000000000000 - 731355101578227875840000000000000 * alpha**2 - 14099310571733416673280000000000 * alpha**4 - 81073030562927817523200000000 * alpha**6 - 3147767878305447936000000 * alpha**8 + 334288184156167127040000 * alpha**10 + 281185408589657644800 * alpha**12 + 86232730330222047 * alpha**14) * np.cos(2 * theta) + 858 * alpha**4 * (4466624150840016896000000000000 + 32365883563431690240000000000 * alpha**2 - 662132097889704345600000000 * alpha**4 - 2398890036552007680000000 * alpha**6 - 1974741212877373440000 * alpha**8 + 323130779332358400 * alpha**10 + 3721112190609957 * alpha**12) * np.cos(4 * theta) + 3458241234987531632640000000000 * alpha**6 * np.cos(6 * theta) + 89376579552028498329600000000 * alpha**8 * np.cos(6 * theta) - 346047349306437402624000000 * alpha**10 * np.cos(6 * theta) - 775073469294637793280000 * alpha**12 * np.cos(6 * theta) - 831787643016281952000 * alpha**14 * np.cos(6 * theta) + 485847822104421777 * alpha**16 * np.cos(6 * theta) - 277601070281549414400000000 * alpha**8 * np.cos(8 * theta) + 37866520993092599808000000 * alpha**10 * np.cos(8 * theta) - 59593413354142187520000 * alpha**12 * np.cos(8 * theta) - 159457746741964070400 * alpha**14 * np.cos( 8 * theta))**2 * np.sin(theta)**4)/(2152766689047092594342740441599811444379 * np.pi * r_g**4 * (6400 + 9 * alpha**2)**26)



def trans_8p_6p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p 6p transition.
    """
    return (1558201490605080576 * G_N * alpha**12 * (3 * (20577034884509550355457095361354072610098380800 + 164428460718120760295785216480697051177287680 * alpha**2 + 476292362176307669225660825102610239324160 * alpha**4 - 1271037637998649554874066388314069401600 * alpha**6 + 3342609904495567715158269886715658240 * alpha**8 + 684659107423441591010122719559680 * alpha**10 - 4738393343265344899070023434240 * alpha**12 + 5163016387839742797284376576 * alpha**14 - 2246765326945569592049664 * alpha**16 + 527063084145229234176 * alpha**18 - 56285002430275584 * alpha**20 + 2996779916641 * alpha**22) + 245 * alpha**2 * (382757345321978243219067994072806410158080 + 2489155709075761298796542777701297029120 * alpha**2 - 17533765948375100642405831554451374080 * alpha**4 + 37052546872163999711389068967280640 * alpha**6 - 29718543043203874161125197086720 * alpha**8 + 6597516053314608088404197376 * alpha**10 + 3726330542106399763070976 * alpha**12 - 2314606695014228557824 * alpha**14 + 479585649026727936 * alpha**16 - 41254207271424 * alpha**18 + 1412376245 * alpha**20) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(2817417536336532260982906125 * np.pi * r_g**4 * (2304 + alpha**2)**28)



def trans_8d_6d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8d 6d transition.
    """
    return (21540577406124633882624 * G_N * alpha**12 * (7 * alpha**2 * (-122055584842304354819175172237975571398656 - 1540199081964397901069761546126463336448 * alpha**2 + 2277718826419616091998591437894582272 * alpha**4 - 3225089575849602267125695106777088 * alpha**6 + 5709405935928278665585420664832 * alpha**8 - 13777539367088385899448238080 * alpha**10 + 12715510802584433313447936 * alpha**12 - 5705719125975534403584 * alpha**14 + 1187453190150291456 * alpha**16 - 114087559590144 * alpha**18 + 3107227739 * alpha**20) * np.cos( 2 * theta) - 3 * (231391925835665474682431846995782499464904704 + 1824244626057615779871173203541699573317632 * alpha**2 + 5364087262578668246695927868144071213056 * alpha**4 - 10224390092000245739511405831241334784 * alpha**6 + 37255485352386591962173821771841536 * alpha**8 - 29006180593859255966069629648896 * alpha**10 + 24322457122053096024456560640 * alpha**12 - 13554535383692356073029632 * alpha**14 + 7850682989261239615488 * alpha**16 - 2296431530104061952 * alpha**18 + 368297811401856 * alpha**20 - 21750594173 * alpha**22 + 1034880 * alpha**4 * (-30415333898162694817609005662208 + 3298665433603381773780878622720 * alpha**2 - 8393607299909438284819857408 * alpha**4 + 7570124600724679682949120 * alpha**6 - 1525110279804435824640 * alpha**8 - 1007763962462208000 * alpha**10 + 569432828805120 * alpha**12 - 94484920320 * alpha**14 + 5764801 * alpha**16) * np.cos( 4 * theta)))**2 * np.sin(theta)**4)/(2727260175173763228631453129 * np.pi * r_g**4 * (2304 + alpha**2)**28)



def trans_8f_6f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8f 6f transition.
    """
    return (6860780745114630269799801815040 * G_N * alpha**12 * (-35752247955613390805249013271309703970816 - 278828862194296446655168709302073425920 * alpha**2 - 782175486997306617543714801721540608 * alpha**4 + 1922021100096709666545627606024192 * alpha**6 - 4567488202553557835300185571328 * alpha**8 + 795910142314097524535721984 * alpha**10 + 2026112300505482441785344 * alpha**12 - 1376932604066930884608 * alpha**14 + 226643281417469952 * alpha**16 + 16010809652992 * alpha**18 + 6107041941 * alpha**20 + 96 * alpha**2 * (-338454816023640253424382525397008384 - 6689543033761711736937030799589376 * alpha**2 - 5244291861560248000094788386816 * alpha**4 + 20118957414600204137797779456 * alpha**6 - 29733283368175352407916544 * alpha**8 + 8513124527686048284672 * alpha**10 - 5836023058105368576 * alpha**12 + 4526625246603264 * alpha**14 - 1719785754148 * alpha**16 + 185061877 * alpha**18) * np.cos(2 * theta) + 33 * alpha**4 * (1105694841871207994234085650202624 - 23064907031248257120591063023616 * alpha**2 + 29444898966870060000438386688 * alpha**4 - 19963934261825140055605248 * alpha**6 + 16000886753035793989632 * alpha**8 - 18718805282732900352 * alpha**10 + 8510084670947328 * alpha**12 - 1618506647296 * alpha**14 + 84119035 * alpha**16) * np.cos(4 * theta) + 31181985048156394659450999275520 * alpha**6 * np.cos(6 * theta) - 220301606894728024559437479936 * alpha**8 * np.cos(6 * theta) + 228420383730347798144483328 * alpha**10 * np.cos(6 * theta) - 50267254015072961298432 * alpha**12 * np.cos(6 * theta) - 37088795913137160192 * alpha**14 * np.cos(6 * theta) + 15925336577998848 * alpha**16 * np.cos(6 * theta) - 1918721540736 * alpha**18 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(2111610439670148205758380411 * np.pi * r_g**4 * (2304 + alpha**2)**28)



def trans_8g_6g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8g 6g transition.
    """
    return (3146677483379767933676220122554429341696 * G_N * alpha**12 * (-483141502817136117865346027163582726144 - 3723503118706261958888172148129529856 * alpha**2 - 8963806473338786312272127597740032 * alpha**4 + 37204016771500286378735837380608 * alpha**6 - 32375270200780775578908229632 * alpha**8 - 3110851757967384539824128 * alpha**10 + 20115215540948696039424 * alpha**12 + 6303410811023523840 * alpha**14 - 665200246556160 * alpha**16 + 207382605430 * alpha**18 + 11 * alpha**2 * (-22696546877598918750614453508636672 - 834798786836459120732011109548032 * alpha**2 - 2384626992221626162962656919552 * alpha**4 + 4044962862133926806176137216 * alpha**6 - 854633626509677370015744 * alpha**8 - 2308095184732529098752 * alpha**10 + 519265947266777088 * alpha**12 + 200108869513728 * alpha**14 + 22734456745 * alpha**16) * np.cos( 2 * theta) + 286 * alpha**4 * (3513016121462506253206702522368 - 35023659851029663454859362304 * alpha**2 - 7622333223751771383398400 * alpha**4 + 20606940951542822338560 * alpha**6 - 6434299281628200960 * alpha**8 - 2029994174251008 * alpha**10 - 3668003759616 * alpha**12 + 981036595 * alpha**14) * np.cos(4 * theta) + 988695931605389841353762930688 * alpha**6 * np.cos(6 * theta) - 4299578672355136935536099328 * alpha**8 * np.cos(6 * theta) + 1249326452709793770504192 * alpha**10 * np.cos(6 * theta) - 162929894833256398848 * alpha**12 * np.cos(6 * theta) + 605462730223583232 * alpha**14 * np.cos(6 * theta) - 432393809143296 * alpha**16 * np.cos(6 * theta) + 42696418765 * alpha**18 * np.cos(6 * theta) + 186616798460916603845345280 * alpha**8 * np.cos(8 * theta) - 479603539351655751352320 * alpha**10 * np.cos(8 * theta) + 80670396492074188800 * alpha**12 * np.cos(8 * theta) + 75861917997465600 * alpha**14 * np.cos(8 * theta) - 28106442524160 * alpha**16 * np.cos( 8 * theta))**2 * np.sin(theta)**4)/(220302205560346892158566069899219 * np.pi * r_g**4 * (2304 + alpha**2)**28)



def trans_8h_6h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8h 6h transition.
    """
    return (3207143635125675681679446857873259568959074598912 * G_N * alpha**12 * (-171783531635057390109821785897697280 - 1304779428651694586337622071705600 * alpha**2 - 2344953837929368388983219814400 * alpha**4 + 20254689360932756707438755840 * alpha**6 + 10715404726611510873292800 * alpha**8 + 533016376385816494080 * alpha**10 + 1208227670719856640 * alpha**12 + 544126480652160 * alpha**14 + 40910500085 * alpha**16 + 364 * alpha**2 * (-20483193542412787854593753088 - 9129789441979229227060297728 * alpha**2 - 47265019761088780176457728 * alpha**4 + 29442978005554581995520 * alpha**6 + 46052972862237573120 * alpha**8 + 11317890655715328 * alpha**10 + 308028839328 * alpha**12 + 288668653 * alpha**14) * np.cos(2 * theta) + 1820 * alpha**4 * (292725377485844515972448256 - 1340427639828487165968384 * alpha**2 - 3696242201487797649408 * alpha**4 - 537963555043934208 * alpha**6 + 1317907689504768 * alpha**8 + 409866791040 * alpha**10 + 33125911 * alpha**12) * np.cos(4 * theta) + 518432335640698810773012480 * alpha**6 * np.cos(6 * theta) - 1035533797790078258380800 * alpha**8 * np.cos(6 * theta) - 1150784061326858649600 * alpha**10 * np.cos(6 * theta) - 256921873473208320 * alpha**12 * np.cos(6 * theta) + 27875392009920 * alpha**14 * np.cos(6 * theta) + 60289158020 * alpha**16 * np.cos(6 * theta) + 145677587221450271490048 * alpha**8 * np.cos(8 * theta) - 166916001736364654592 * alpha**10 * np.cos(8 * theta) - 110199678164140032 * alpha**12 * np.cos(8 * theta) - 37427862955392 * alpha**14 * np.cos(8 * theta) + 9043373703 * alpha**16 * np.cos(8 * theta) + 11937439186786713600 * alpha**10 * np.cos(10 * theta) - 8289888324157440 * alpha**12 * np.cos(10 * theta) - 5846817936960 * alpha**14 * np.cos( 10 * theta))**2 * np.sin(theta)**4)/(55774526757540776327083759783733677 * np.pi * r_g**4 * (2304 + alpha**2)**28)



def trans_8p_7p(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8p 7p transition.
    """
    return (1272155426137788072853504 * G_N * alpha**12 * (58586326592779097385262512100031491947155375257714421829468160 + 85665629033797635641678950791439486812013142172252755722240 * alpha**2 - 45949254755723598249311019156064835000815432992433373184 * alpha**4 + 48751439694662808827097493846936264505982195775569920 * alpha**6 - 2711377150476710287242396188563058986966046474240 * alpha**8 - 4356456355080110741075338394373870646387015680 * alpha**10 + 2052845426380902102184073132710068507115520 * alpha**12 - 364126052148002830327253358999846780928 * alpha**14 + 35653632552055079027808424427520000 * alpha**16 - 1901112719969387161269043200000 * alpha**18 + 57883411318227747840000000 * alpha**20 - 876303900391500000000 * alpha**22 + 6482895908203125 * alpha**24 + 525 * alpha**2 * (66720944097097186344367839035203503037485622332491825152 - 116660173473282547331736750145597704768509036527091712 * alpha**2 + 83407991106376570342928081392662827194216276295680 * alpha**4 - 27488905818660001434988139208106881236955299840 * alpha**6 + 4400658636859266931656031175082649300500480 * alpha**8 - 269668792691516930460614031475784810496 * alpha**10 - 9723504452565624879493844391952384 * alpha**12 + 2334319589660352468769308672000 * alpha**14 - 130241165138207598182400000 * alpha**16 + 3349507669079040000000 * alpha**18 - 39703891500000000 * alpha**20 + 192216796875 * alpha**22) * np.cos( 2 * theta))**2 * np.sin(theta)**4)/(177547277067485265433788299560546875 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_8d_7d(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8d 7d transition.
    """
    return (195484491402037066426960838656 * G_N * alpha**12 * (474094887027756024775528168880917214051097034652649152053248 + 765547180975803704232861006442129555579292742963316129792 * alpha**2 - 418023037549674005399578848461734828742258465854980096 * alpha**4 + 530388301130355248962866614023056153346476909527040 * alpha**6 - 173776901892328797108385744197156449814157721600 * alpha**8 + 46000909981498208505223088622884125811933184 * alpha**10 - 7133818282931338665469174224017965449216 * alpha**12 + 979266639507202585848940673229651968 * alpha**14 - 91630539453704176800417762508800 * alpha**16 + 5943062581348425667706880000 * alpha**18 - 215981508474463334400000 * alpha**20 + 4329898029630000000 * alpha**22 - 34253033203125 * alpha**24 - alpha**2 * (-368289442974505666244326897450370159147501823329690451968 + 211889691759953376042341833401443182624251898811645952 * alpha**2 - 120113010447067025519881341494817845236505342115840 * alpha**4 + 48004863156887166482433774214192533476894310400 * alpha**6 - 28913337158320612221289657385636405339750400 * alpha**8 + 9452149812349901284066493856878715469824 * alpha**10 - 1673395798852833342698108121247645696 * alpha**12 + 157750284799542519262887070924800 * alpha**14 - 8166696422479011690577920000 * alpha**16 + 219692353343081472000000 * alpha**18 - 2842647990840000000 * alpha**20 + 11417677734375 * alpha**22) * np.cos(2 * theta) + 3104640 * alpha**4 * (25037170419141229323928169704442721475563094016 - 25052914584721763017307533133394593657651200 * alpha**2 + 9892489028070038479975353346356380958720 * alpha**4 - 1687825112439057817304572028264644608 * alpha**6 + 104618085900432442014782623580160 * alpha**8 + 4470624546666395748827922432 * alpha**10 - 911511778202651669299200 * alpha**12 + 45773361961205760000 * alpha**14 - 949072269600000 * alpha**16 + 7688671875 * alpha**18) * np.cos( 4 * theta))**2 * np.sin(theta)**4)/(928075126687158979475498199462890625 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_8f_7f(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8f 7f transition.
    """
    return (3937262487050993503493489991514270466048 * G_N * alpha**12 * (479506824171307584063486618876968716175129321709074644992 + 818900340482354302827000913019847766890778953963798528 * alpha**2 - 571140987309513813505595563477790706368425161129984 * alpha**4 + 557082544151299407660747802532029264395459624960 * alpha**6 - 132758018936154810734210120472776619977605120 * alpha**8 + 10343676655160644880003603262086961954816 * alpha**10 + 2587764208169803848658749007494905856 * alpha**12 - 474787815698623467142703615049728 * alpha**14 + 28380676468748528149069824000 * alpha**16 - 563316828562175385600000 * alpha**18 + 163767353904000000 * alpha**20 - 399111268359375 * alpha**22 - 96 * alpha**2 * (-4273419396902994526579604838476680874572260254416896 - 691578960437634207005836377231136547039342493696 * alpha**2 + 1480279009002706710377251232795666163538329600 * alpha**4 - 1039710377903464670207837274403386419052544 * alpha**6 + 214371942890899873446903299576199905280 * alpha**8 - 27579551415206806167770346440097792 * alpha**10 + 2827513975919181653703003209728 * alpha**12 - 347364847858096870726041600 * alpha**14 + 25433665525169112960000 * alpha**16 - 930386761530187500 * alpha**18 + 12094280859375 * alpha**20) * np.cos(2 * theta) - 33 * alpha**4 * (-3735838231550111505993390642482252002520727552000 + 2316534424456846912733896379276613675059773440 * alpha**2 - 734669960233099461980652498270067571556352 * alpha**4 + 156802177347523518021324055566656471040 * alpha**6 - 40944971533163273181264720596828160 * alpha**8 + 7319235103138154399927747215360 * alpha**10 - 711442067634395206700236800 * alpha**12 + 33972956223771156480000 * alpha**14 - 755125365456000000 * alpha**16 + 5497400390625 * alpha**18) * np.cos(4 * theta) + 8769924893760090117562057938587583764653670400 * alpha**6 * np.cos( 6 * theta) - 4952040374465286841267354228386054159728640 * alpha**8 * np.cos( 6 * theta) + 1017690465705818952336866124164958781440 * alpha**10 * np.cos( 6 * theta) - 63477319030386544459933090106572800 * alpha**12 * np.cos( 6 * theta) - 3124405801321944438707689881600 * alpha**14 * np.cos(6 * theta) + 564407285509719604592640000 * alpha**16 * np.cos(6 * theta) - 22610872483418419200000 * alpha**18 * np.cos(6 * theta) + 318592384110000000 * alpha**20 * np.cos( 6 * theta))**2 * np.sin(theta)**4)/(15527624944602856885604560375213623046875 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_8g_7g(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8g 7g transition.
    """
    return (485716125904076626662528756232132919096855654039552 * G_N * alpha**12 * (14847796802012240464949856983323538536822456664981504 + 26189728032471925721007552819390812440042454122496 * alpha**2 - 25009366530475821836477742363313594075812200448 * alpha**4 + 16193112567427768099735915858942651971665920 * alpha**6 - 3190752318779093754683957307948180439040 * alpha**8 + 24864369135344874809139544952143872 * alpha**10 + 32823906007768659299492106862592 * alpha**12 - 1583359984455655268722671616 * alpha**14 - 328292085720379903180800 * alpha**16 + 7413805094008320000 * alpha**18 - 196769470218750 * alpha**20 - 3 * alpha**2 * (-4286824842384671860135025600552666121962239557632 - 3763497379483152502491962241889071673966067712 * alpha**2 + 3672291059687189639751328247682713203507200 * alpha**4 - 1341776781560836925964470733388128452608 * alpha**6 + 74520015540763174001649948179824640 * alpha**8 + 18749706269344586438965708455936 * alpha**10 - 2332932423216897791186960384 * alpha**12 + 20227437227039511347200 * alpha**14 + 3097314460503648000 * alpha**16 + 79093610578125 * alpha**18) * np.cos(2 * theta) - 78 * alpha**4 * (-55988705818778715462842275603311195341193216 + 10766365197710959099056792431977964765184 * alpha**2 + 2439201897108913354354282474103635968 * alpha**4 - 1389013153975367040848772045209600 * alpha**6 + 93221790868654808481272954880 * alpha**8 - 6178368345847128184061952 * alpha**10 + 1424965164211472629760 * alpha**12 - 139629729941856000 * alpha**14 + 3413045109375 * alpha**16) * np.cos(4 * theta) + 492414326844315307984845268997307396784128 * alpha**6 * np.cos( 6 * theta) - 166145827784359102309820455328503824384 * alpha**8 * np.cos( 6 * theta) + 24307472782076731014705120301547520 * alpha**10 * np.cos( 6 * theta) - 3104456776903835742812409692160 * alpha**12 * np.cos(6 * theta) + 704054452008962631967703040 * alpha**14 * np.cos(6 * theta) - 73607038832878118830080 * alpha**16 * np.cos(6 * theta) + 3194194312397088000 * alpha**18 * np.cos(6 * theta) - 40511361515625 * alpha**20 * np.cos(6 * theta) + 14857675527174697636110033674399907840 * alpha**8 * np.cos( 8 * theta) - 4120650742029213920279961576407040 * alpha**10 * np.cos( 8 * theta) + 302754878829100663225240780800 * alpha**12 * np.cos(8 * theta) + 17744074593186321373593600 * alpha**14 * np.cos(8 * theta) - 2347302904064262144000 * alpha**16 * np.cos(8 * theta) + 67756602513600000 * alpha**18 * np.cos( 8 * theta))**2 * np.sin(theta)**4)/(1818161147974715438853241503238677978515625 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_8h_7h(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8h 7h transition.
    """
    return (7669740062953234774809001582573794982362846873820808521443180544 * G_N * alpha**12 * (-25904463071312308846856934585776020730717143040 - 46464478563817518260067046251591236163993600 * alpha**2 + 60812164990873816357417421469457349345280 * alpha**4 - 18974380877057697664868726164632371200 * alpha**6 + 2593075354048310428143465749544960 * alpha**8 + 505933520339108381696710410240 * alpha**10 - 4152107781642266240614400 * alpha**12 + 154351392563790643200 * alpha**14 + 81977920879344000 * alpha**16 + 1207635024375 * alpha**18 + 52 * alpha**2 * (-410576498812561446932247741087729663868928 - 717537828047980154591964240828419801088 * alpha**2 + 496893376998350769894822661167513600 * alpha**4 - 78079405640345804367642989953024 * alpha**6 - 6374966245349435082685808640 * alpha**8 + 1393057139448519730397184 * alpha**10 + 93766800183066959872 * alpha**12 - 597570705362400 * alpha**14 + 59648369625 * alpha**16) * np.cos( 2 * theta) + 5460 * alpha**4 * (-1354129491738219004021775976326234112 - 458406305780090154003274760454144 * alpha**2 + 254546896761089255229987225600 * alpha**4 - 15689089247373725355999232 * alpha**6 - 5104783320013476986880 * alpha**8 + 204535742562729984 * alpha**10 + 20247511303040 * alpha**12 + 325947375 * alpha**14) * np.cos( 4 * theta) - 988783596521374448231863969634058240 * alpha**6 * np.cos( 6 * theta) + 37706523307815869156765466624000 * alpha**8 * np.cos(6 * theta) + 22330479133186963537416683520 * alpha**10 * np.cos(6 * theta) - 1742453062802031614361600 * alpha**12 * np.cos(6 * theta) - 178053161682512363520 * alpha**14 * np.cos(6 * theta) - 24315432770865600 * alpha**16 * np.cos(6 * theta) + 1779672667500 * alpha**18 * np.cos(6 * theta) - 48144606817136698550944228442112 * alpha**8 * np.cos(8 * theta) + 4512541787618120065949368320 * alpha**10 * np.cos(8 * theta) - 108051490007737234882560 * alpha**12 * np.cos(8 * theta) + 59988818077752066048 * alpha**14 * np.cos(8 * theta) - 10761469281763200 * alpha**16 * np.cos(8 * theta) + 266950900125 * alpha**18 * np.cos(8 * theta) - 737117678547689611104092160 * alpha**10 * np.cos(10 * theta) + 53903039957926045286400 * alpha**12 * np.cos(10 * theta) + 3480947134781276160 * alpha**14 * np.cos(10 * theta) - 438511345272000 * alpha**16 * np.cos( 10 * theta))**2 * np.sin(theta)**4)/(103334662930761306388378031551837921142578125 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_8i_7i(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 8i 7i transition.
    """
    return (1655796070000806022650404428924209290585059099412269658845972066401620328448 * G_N * alpha**12 * (-1049303679634552282513864135533464754585600 - 1891962713926908353778608756530505318400 * alpha**2 + 3330671415718122013979940068995891200 * alpha**4 + 76687273953109312808316567552000 * alpha**6 + 2394959565019148609126400000 * alpha**8 + 5763832700958281784360960 * alpha**10 + 355327029831403929600 * alpha**12 + 1851148458036864 * alpha**14 + 156705468750 * alpha**16 + 450 * alpha**2 * (-1722202730492634392256210831692267520 - 5099710396436865275639520653475840 * alpha**2 + 2070062769991297542279030374400 * alpha**4 + 319565881073409808568156160 * alpha**6 + 8103823671469686128640 * alpha**8 + 390344056966348800 * alpha**10 + 43635082938496 * alpha**12 + 515386875 * alpha**14) * np.cos( 2 * theta) + 30600 * alpha**4 * (-8225963710663685069632595558400 - 8676078265322256247396761600 * alpha**2 + 1041367828281010264473600 * alpha**4 + 313969544804821893120 * alpha**6 + 14967572709734400 * alpha**8 + 96459408248 * alpha**10 + 8685375 * alpha**12) * np.cos(4 * theta) - 34575409490174011104622018560000 * alpha**6 * np.cos(6 * theta) - 12413743757282498146467840000 * alpha**8 * np.cos(6 * theta) + 86491349881294159872000 * alpha**10 * np.cos(6 * theta) + 209185405801021440000 * alpha**12 * np.cos(6 * theta) + 11598204229852800 * alpha**14 * np.cos(6 * theta) + 148556784375 * alpha**16 * np.cos(6 * theta) - 1971861332257849592512512000 * alpha**8 * np.cos(8 * theta) - 307241012481188954112000 * alpha**10 * np.cos(8 * theta) - 8171607181946880000 * alpha**12 * np.cos(8 * theta) + 1048343660067200 * alpha**14 * np.cos(8 * theta) + 139154456250 * alpha**16 * np.cos(8 * theta) - 48330367488281511198720 * alpha**10 * np.cos(10 * theta) - 4596321889586380800 * alpha**12 * np.cos(10 * theta) - 259490057588608 * alpha**14 * np.cos(10 * theta) + 20685121875 * alpha**16 * np.cos(10 * theta) - 423129716545536000 * alpha**12 * np.cos(12 * theta) - 33731641944000 * alpha**14 * np.cos( 12 * theta))**2 * np.sin(theta)**4)/(59527962117316694764583953656256198883056640625 * np.pi * r_g**4 * (12544 + alpha**2)**30)



def trans_9k_8k(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 9k 8k transition.
    """
    return (80630095298842303429095905162840439812164034993325871766922748866571777319936650504304590848 * G_N * alpha**12 * (272 * alpha**2 * (-426408178309884282694390525056710333882695680 - 1179633100740962894563268093748704056442880 * alpha**2 + 449226001100280974566956280953522094080 * alpha**4 + 22529659090090443298732151006035968 * alpha**6 - 247988244614590825378363736064 * alpha**8 + 48890273341543820147294208 * alpha**10 + 2794715535865742346240 * alpha**12 + 5489255572409160 * alpha**14 + 269651341625 * alpha**16) * np.cos(2 * theta) + 3 * (-75405042780068769292729633336159218247801440829440 - 93590262344292225532368803558981729519916810240 * alpha**2 + 154615184626186572798871841968967840169984000 * alpha**4 - 7897290923304621601224453499111751024640 * alpha**6 + 344880326147933093874852715875532800 * alpha**8 + 127705922637348764238333062676480 * alpha**10 + 2871807017489961510701629440 * alpha**12 - 2358320717223552614400 * alpha**14 + 1663621228621900800 * alpha**16 + 10133213574750 * alpha**18 + 1615 * alpha**4 * (-5822009164015650550954972756828634480640 - 6670132502121819989932312012951388160 * alpha**2 + 947722542844296990215226569785344 * alpha**4 + 111189382825001011752141324288 * alpha**6 + 1935030641553189554356224 * alpha**8 + 10852598726891274240 * alpha**10 + 1760067649774080 * alpha**12 + 10905566725 * alpha**14) * np.cos(4 * theta) + 90440 * alpha**6 * (-12028802211687012130367797513420800 - 4814959347003467865491238813696 * alpha**2 + 203744298899173202416631808 * alpha**4 + 49405753216607100862464 * alpha**6 + 1476604041406156800 * alpha**8 + 3868489046160 * alpha**10 + 209147855 * alpha**12) * np.cos( 6 * theta) - 55494044896796630756406039165272064 * alpha**8 * np.cos( 8 * theta) - 9044847570842148420248307499008 * alpha**10 * np.cos( 8 * theta) - 29105518831818076726493184 * alpha**12 * np.cos(8 * theta) + 44486913001120951173120 * alpha**14 * np.cos(8 * theta) + 1538903752947348480 * alpha**16 * np.cos(8 * theta) + 10403432603410 * alpha**18 * np.cos(8 * theta) - 1353067683365876139561729392640 * alpha**10 * np.cos( 10 * theta) - 111750830558467022955479040 * alpha**12 * np.cos(10 * theta) - 1653002446709093990400 * alpha**14 * np.cos(10 * theta) + 108096589932312960 * alpha**16 * np.cos(10 * theta) + 9341857847960 * alpha**18 * np.cos(10 * theta) - 15562320652089508296130560 * alpha**12 * np.cos(12 * theta) - 864868991715246735360 * alpha**14 * np.cos(12 * theta) - 28217287965335040 * alpha**16 * np.cos(12 * theta) + 1380047182085 * alpha**18 * np.cos(12 * theta) - 67871980316838297600 * alpha**14 * np.cos(14 * theta) - 3273147198921600 * alpha**16 * np.cos( 14 * theta)))**2 * np.sin(theta)**4)/(763103482836293187320433699957796773008680132070767507225 * np.pi * r_g**4 * (20736 + alpha**2)**34)



def trans_13k_8k(alpha, theta, G_N=1.0, r_g=1.0):
    """
    Differential power per solid angle for 13k 8k transition.
    """
    return (11151418536848547755867924319885853221767819952788222916784887502410179051038741619730355847168000000000000 * G_N * alpha**12 * (20308114472669529824659093984268161047813954593425943578648547787534237696 + 263580360125485651981576847309732426399397434547820714001432279110385664 * alpha**2 + 1360396709912590172207287835094004068927232107323887499859103630491648 * alpha**4 + 736625864191384301689549358831733145564659396710749885091649945600 * alpha**6 - 31418713731528447057018492392828306179417955496513969202397184000 * alpha**8 - 166813509597171007917426577092480234434954710006118744064000000 * alpha**10 - 254536737471005308482459960053148769425096228030382080000000 * alpha**12 + 186339300384291128215117034496933030606617444352000000000 * alpha**14 + 64686217657221579665384646261080506918502400000000000 * alpha**16 - 164632922216790019555567271785581772800000000000000 * alpha**18 - 46348416852801048984067504081920000000000000000 * alpha**20 + 5360455058308814813500242000000000000000000 * alpha**22 - 1799385467848646301834082031250000000000 * alpha**24 - 119100120359000461363792419433593750 * alpha**26 - 816 * alpha**2 * (52498665785608492872888818121021783306656626526770926727251250642944 + 260242963510832678294306327467416682829621981190690295125265874944 * alpha**2 - 4223077819568805090910948755703630398603750995638405587114393600 * alpha**4 - 50210900334131812500533220327973295244217695928710754782412800 * alpha**6 - 184253356454023768592334630909318750806421911978356244480000 * alpha**8 - 56711783160614153428789041742181350613043401195520000000 * alpha**10 + 870619262311929401740823567917411991619108864000000000 * alpha**12 + 458262245559960639521364208038981088051200000000000 * alpha**14 - 129739195542175128626329406599004160000000000000 * alpha**16 + 10335382249008061282499538336000000000000000 * alpha**18 + 43342671279298614987120032812500000000000 * alpha**20 - 346480187558195631839447021484375000 * alpha**22 + 352147881546997940540313720703125 * alpha**24) * np.cos( 2 * theta) - 4845 * alpha**4 * (14467969927662380582888228762339864696092321431351723850163290112 + 286431118908974883982553618868671691782552148090341567968051200 * alpha**2 + 1639084988547998401565291512390605031079809276125784100044800 * alpha**4 - 4429257827872154057445236379582612845614246207160320000 * alpha**6 - 26121088455876539609241402179374165046734010449920000000 * alpha**8 - 55271027402783717691205277769288478044979200000000000 * alpha**10 + 20909151698899445176378824313890511257600000000000 * alpha**12 + 49933742017266394134795963004354560000000000000 * alpha**14 + 9646793764151849018073245184000000000000000 * alpha**16 - 1253312885792658854381400000000000000000 * alpha**18 + 631851490384359475166015625000000000 * alpha**20 + 42725975655840747356414794921875 * alpha**22) * np.cos( 4 * theta) + 28784831292101228394543378057570353340475828187995908102684672000 * alpha**6 * np.cos(6 * theta) - 487855545028957731387388635423400512857778038518730295607296000 * alpha**8 * np.cos(6 * theta) - 8832840070425990415459766360334036327192342876223281561600000 * alpha**10 * np.cos(6 * theta) - 27302458458444204447306037855340668001753458173542400000000 * alpha**12 * np.cos(6 * theta) + 37225093772847894965953426756657069097232629760000000000 * alpha**14 * np.cos(6 * theta) + 105082414645811771461060230490394820870144000000000000 * alpha**16 * np.cos(6 * theta) + 35401419501654863259107376264039628800000000000000 * alpha**18 * np.cos(6 * theta) - 36445335431476457029237945305600000000000000000 * alpha**20 * np.cos(6 * theta) - 18442210903120353373234972125000000000000000 * alpha**22 * np.cos( 6 * theta) + 267298336323422384379235839843750000000 * alpha**24 * np.cos( 6 * theta) - 222320224670134194545745849609375000 * alpha**26 * np.cos( 6 * theta) + 48937762679557201068897497950102832050366125834245677409894400 * alpha**8 * np.cos(8 * theta) + 524316663077761637230981511560704612412801828030292951040000 * alpha**10 * np.cos(8 * theta) - 1820443350370860922588808023133022352842482201395200000000 * alpha**12 * np.cos(8 * theta) - 19981067169717879195215290683311919558316523520000000000 * alpha**14 * np.cos(8 * theta) + 7486839350823729685230312983785785996083200000000000 * alpha**16 * np.cos(8 * theta) + 20882088763590809299852809939116359680000000000000 * alpha**18 * np.cos(8 * theta) + 9073537964487046210307073518592000000000000000 * alpha**20 * np.cos( 8 * theta) - 1681940964249499117108344400000000000000000 * alpha**22 * np.cos( 8 * theta) - 1619015737871258945949199218750000000000 * alpha**24 * np.cos( 8 * theta) - 122276123568573807000160217285156250 * alpha**26 * np.cos( 8 * theta) + 4279726794133251688033137978501832979725793134588723200000 * alpha**10 * np.cos(10 * theta) + 274217878674425487235402853291222489722737823580160000000 * alpha**12 * np.cos(10 * theta) + 886216741031685427764605986844469561095356416000000000 * alpha**14 * np.cos(10 * theta) - 6183554493553550520327169121981534358732800000000000 * alpha**16 * np.cos(10 * theta) + 530296735766588076733647295532236800000000000000 * alpha**18 * np.cos(10 * theta) + 3246384883419627368709161304576000000000000000 * alpha**20 * np.cos( 10 * theta) + 453378744194533354785012425000000000000000 * alpha**22 * np.cos( 10 * theta) + 221955072619501389559504394531250000000 * alpha**24 * np.cos( 10 * theta) - 109798968102392806285858154296875000 * alpha**26 * np.cos( 10 * theta) - 2946232812343198522804283954144794454077042851840000000 * alpha**12 * np.cos(12 * theta) + 21982065020272651644862723498396473190514688000000000 * alpha**14 * np.cos(12 * theta) + 352477455279600166717198106303172653875200000000000 * alpha**16 * np.cos(12 * theta) - 826864506850977095831442066898944000000000000000 * alpha**18 * np.cos(12 * theta) - 66090924956068470602512642560000000000000000 * alpha**20 * np.cos( 12 * theta) + 216344526096361207258513800000000000000000 * alpha**22 * np.cos( 12 * theta) + 99558399014565674982626953125000000000 * alpha**24 * np.cos( 12 * theta) - 16220302106035300928592681884765625 * alpha**26 * np.cos( 12 * theta) - 318932403729334962497721672103576888934400000000000 * alpha**14 * np.cos(14 * theta) - 2131201398275388748548882372734484480000000000000 * alpha**16 * np.cos(14 * theta) + 31217051385212967758981552799744000000000000000 * alpha**18 * np.cos(14 * theta) - 39333539834364144059893762560000000000000000 * alpha**20 * np.cos( 14 * theta) - 11038630958812796933719125000000000000000 * alpha**22 * np.cos( 14 * theta) + 12995465746583541840270996093750000000 * alpha**24 * np.cos( 14 * theta))**2 * np.sin(theta)**4)/(3345612719303356849704192295086162487740759683482583878860085745358729 * np.pi * r_g**4 * (43264 + 25 * alpha**2)**42)




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
    "6h": ann_6h,
    "7p": ann_7p,
    "7d": ann_7d,
    "7f": ann_7f,
    "7g": ann_7g,
    "7h": ann_7h,
    "7i": ann_7i,
    "8p": ann_8p,
    "8d": ann_8d,
    "8f": ann_8f,
    "8g": ann_8g,
    "8h": ann_8h,
    "8i": ann_8i,
    "8k": ann_8k
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
    "7p 6p": trans_7p_6p,
    "7d 6d": trans_7d_6d,
    "7g 6g": trans_7g_6g,
    "7h 6h": trans_7h_6h,
    "8p 2p": trans_8p_2p,
    "8p 3p": trans_8p_3p,
    "8d 3d": trans_8d_3d,
    "8p 4p": trans_8p_4p,
    "8d 4d": trans_8d_4d,
    "8f 4f": trans_8f_4f,
    "8p 5p": trans_8p_5p,
    "8d 5d": trans_8d_5d,
    "8f 5f": trans_8f_5f,
    "8g 5g": trans_8g_5g,
    "8p 6p": trans_8p_6p,
    "8d 6d": trans_8d_6d,
    "8f 6f": trans_8f_6f,
    "8g 6g": trans_8g_6g,
    "8h 6h": trans_8h_6h,
    "8p 7p": trans_8p_7p,
    "8d 7d": trans_8d_7d,
    "8f 7f": trans_8f_7f,
    "8g 7g": trans_8g_7g,
    "8h 7h": trans_8h_7h,
    "8i 7i": trans_8i_7i,
    "9k 8k": trans_9k_8k,
    "13k 8k": trans_13k_8k
}
