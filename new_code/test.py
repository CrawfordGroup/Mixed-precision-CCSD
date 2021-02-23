# Input

# settings (psi4, mol, field, propagation, plots)
# ground state CCSD
# propagation
# spectral property

import psi4
from psi4 import constants as pc
import numpy as np
from ccenergy import ccEnergy
from cclambda import ccLambda
from ccenergy_dp import ccEnergy_dp
from cclambda_dp import ccLambda_dp
from cclambda_sp import ccLambda_sp
from ccpert import CCPert, CCLinresp
from ccpert_dp import CCPert_dp, CCLinresp_dp
#from coordinate import Mol

np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

## Setup and parameters

# Psi4 setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
numpy_memory = 2

### Water molecule

#mol = psi4.geometry(
"""
O
H 1 1.1
H 1 1.1 2 104
symmetry c1
"""
#)

### Water cluster
## Number of water molecules
## 2

#mol = psi4.geometry(
"""
O -1.5167088799 -0.0875022822  0.0744338901
H -0.5688047242  0.0676402012 -0.0936613229
H -1.9654552961  0.5753254158 -0.4692384530
O  1.3898685804  0.0960995460 -0.0761488482
H  1.5926924704 -0.8335878302 -0.2679884752
H  1.5164596797  0.1745974125  0.8831816344
symmetry c1
"""
#)

## 3

mol = psi4.geometry(
"""
O -1.4765014766 -0.6332885087  0.0898827346
H -1.9838499390 -0.7470663319 -0.7281466521
H -1.1474120728  0.2937933586  0.0499611499
O  1.3009060824 -0.9725326095  0.1123300788
H  1.6231813194 -1.4263451978 -0.6812347553
H  0.3383752926 -1.1745452783  0.1460364674
O  0.2066022512  1.5796259613 -0.1194070925
H  0.8505520801  0.8353634930 -0.0913819530
H  0.3749571446  2.0763034073  0.6962042911
symmetry c1
"""
)

## 4

#mol = psi4.geometry(
"""
O -1.9292712102 -0.0645569159  0.1117679206
H -2.4914119096  0.0073852098 -0.6743533999
H -1.3279797920  0.7200450587  0.0572351830
O  0.0556775311 -1.9834041949  0.1251954700
H -0.7452932853 -1.4024421984  0.1534116762
H -0.0578054438 -2.5263517257 -0.6694026790
O -0.0284527051  1.9145766189 -0.1269877741
H  0.0826356714  2.4643236809  0.6632720821
H  2.4967666426 -0.0808647657  0.7228603336
O  1.9591510378 -0.0026698519 -0.0796304864
H  0.7727195176  1.3336367194 -0.1473249006
H  1.3542775597 -0.7856500486 -0.0462427112
symmetry c1
"""
#)

## 5

#mol = psi4.geometry(
"""
O -0.2876645445 -1.7012437583  0.2029164243
H -0.5715171913 -2.6278936014  0.2177698265
H -1.1211952210 -1.1567502352  0.1410434902
O  1.4494811981 -0.1832356135 -1.3664308071
H  0.8495877249 -0.8784465376 -1.0148018507
H  1.7885868060 -0.5222367118 -2.2083896644
O -2.3074119608  0.0600951514 -0.0549593403
H -2.8203525262  0.2892204097  0.7353401875
H  0.2626613561  1.4965603247  0.5922725167
O -0.2424726136  1.8126064878 -0.1909073151
H -1.6640497911  0.8139612893 -0.1634430169
H  0.2878907915  1.4335334111 -0.9188689235
O  1.3813546468  0.2415962603  1.5136093974
H  0.8356415445 -0.5611886184  1.3988328403
H  1.9779314005  0.1892600736  0.7452520171
symmetry c1
"""
#)

## 6

#mol = psi4.geometry(
"""
O -1.0056893157 -0.1043674637  1.7352314342
H -1.0664276447 -0.3762387797  2.6634016204
H -1.5071027077 -0.7913788396  1.2185432381
O  1.1470564813  0.3117796882 -0.0477367962
H  0.6103606068  0.0433421321  0.7275681882
H  2.0791424333  0.1211528401  0.1794662548
O -2.2063795158 -1.8574899369  0.0516378330
H -3.1585603577 -1.7418528352 -0.0901912407
H -1.2703988826  0.5556485786 -1.4822595530
O -0.9686544539 -0.3213451857 -1.8092615790
H -1.7829183163 -1.4200976013 -0.7381133801
H -0.0506196244 -0.3301023162 -1.4666019324
O -1.1414038621  2.0193691143 -0.2156011398
H -1.3619510638  1.5449391020  0.6091355621
H -0.1726333256  1.9018225581 -0.2491925658
O  3.9130618957 -0.1477904028  0.5773094099
H  4.3921274685  0.6778964502  0.4038875571
H  4.3084563274 -0.7931325208 -0.0300166098
symmetry c1
"""
#)

## 7

#mol = psi4.geometry(
"""
O -0.2578815727  0.8589987101  1.0903178485
H  0.5724321179  0.9957545630  1.6134194631
H -0.3379575048 -0.1213199214  1.0257127358
O  0.7053437993  1.0245432710 -1.4943936709 
H  0.3698578944  1.1189598601 -0.5656693305
H  1.0352513822  1.9002918475 -1.7483588311
O -0.3458650799 -1.7904919129  0.2521395252
H -0.8067013063 -2.5793300144  0.5772833215
H -2.2204485019  0.0514045678 -1.2281883065
O -1.6470360380 -0.5567512801 -1.7481190861
H -0.8635215557 -1.4793338112 -0.5471909111
H -0.9032735713  0.0137100383 -2.0286082118
O -2.8270840011  1.2361544476  0.0947724694
H -2.0207774964  1.2912094835  0.6512402118
H -3.0198162448  2.1508831339 -0.1596488039
O  2.3155522081  0.8259479989  1.9468048976
H  2.6243054144  0.3877453957  2.7538641108
H  2.4860312413  0.1745463055  1.2179450536
O  2.3555214451 -0.9273649757 -0.1787945084
H  1.5622606691 -1.4797809021 -0.0212223033
H  2.0415014395 -0.3067984167 -0.8674027785
symmetry c1
"""
#)

# Uracil

#mol = psi4.geometry(
"""
0 1
C  1.1965439001  1.1059975965  0.0000000000
C -0.0106406622  1.6974586473  0.0000000000
N -1.1759657685  0.9706327770  0.0000000000
C  1.2905520182 -0.3511811123  0.0000000000
N  0.0394261039 -0.9922273002  0.0000000000
C -1.2164849902 -0.4175249683  0.0000000000
O -2.2534478773 -1.0446838400  0.0000000000
O  2.3153668717 -1.0016569295  0.0000000000
H  2.1145824324  1.6760396163  0.0000000000
H -0.1369773173  2.7740930054  0.0000000000
H -2.0769371453  1.4242304202  0.0000000000
H  0.0555272212 -2.0045027192  0.0000000000
symmetry c1
"""
#)

# Benzene
#mol = psi4.geometry(
"""
0 1
C  1.4059535336  0.0000000000 0.0000000000
C  0.7029767668  1.2175914766 0.0000000000
C  0.7029767668 -1.2175914766 0.0000000000 
C -0.7029767668  1.2175914766 0.0000000000
C -0.7029767668 -1.2175914766 0.0000000000
C -1.4059535336  0.0000000000 0.0000000000 
H  2.5018761059  0.0000000000 0.0000000000 
H  1.2509380530  2.1666882648 0.0000000000
H  1.2509380530 -2.1666882648 0.0000000000 
H -1.2509380530  2.1666882648 0.0000000000
H -1.2509380530 -2.1666882648 0.0000000000
H -2.5018761059  0.0000000000 0.0000000000 
symmetry c1
"""
#)

# Chiral molecules
# (S)-1,3-dimethylallene
mol = psi4.geometry(
"""
C  0.000000  0.000000  0.414669
C  0.000000  1.309935  0.412904
C  0.000000 -1.309935  0.412904
H -0.677576  1.833097  1.091213
H  0.677576 -1.833097  1.091213
C  0.875978  2.177784 -0.460206
C -0.875978 -2.177784 -0.460206
H  1.519837  1.571995 -1.104163
H -1.519837 -1.571995 -1.104163
H  0.267559  2.832375 -1.098077
H -0.267559 -2.832375 -1.098077
H  1.514280  2.829499  0.150828
H -1.514280 -2.829499  0.150828
symmetry c1
"""
)

# (S)-2-chloropropionitrile
#mol = psi4.geometry(
"""
C  -0.074850  0.536809  0.165803
C  -0.131322  0.463480  1.692375
H   0.576771  1.182613  2.118184
C   1.260194  0.235581 -0.354931
N   2.332284  0.017558 -0.742030
CL -1.285147 -0.612976 -0.578450
H  -0.370843  1.528058 -0.186624
H  -1.138978  0.710124  2.035672
H   0.130424 -0.538322  2.041140
symmetry c1
"""
#)

# (R)-methylthiirane
#mol = psi4.geometry(
"""
C    0.364937   1.156173  -0.153047
C   -0.521930   0.166108   0.502066
C   -1.785526  -0.309717  -0.178180
H   -1.645392  -0.376658  -1.261546
S    1.101404  -0.526707  -0.054082
H    0.106264   1.485766  -1.157365
H    0.850994   1.912823   0.458508
H   -0.587772   0.237291   1.586932
H   -2.083855  -1.297173   0.188561
H   -2.607593   0.389884   0.025191
symetry c1
"""
#)

# (S)-methyloxirane:
#mol = psi4.geometry(
"""
 zmat = (
 C
 C 1 B1
 O 2 B2 1 A1
 H 2 B3 1 A2 3 D1
 H 2 B4 1 A3 3 D2
 H 1 B5 2 A4 3 D3
 C 1 B6 2 A5 3 D4
 H 7 B7 1 A6 2 D5
 H 7 B8 1 A7 2 D6
 H 7 B9 1 A8 2 D7
  )
  zvars = (
 A1 59.24053923
 A2 119.43315591
 A3 120.16908378
 A4 117.21842257
 A5 122.37833117
 A6 110.6489744
 A7 110.71820821
 A8 110.62534492
 D1 -103.61051046
 D2 103.47749463
 D3 -102.30478838
 D4 103.75862518
 D5 -24.51750626
 D6 96.01547667
 D7 -144.34513611
 B1 1.47013445
 B2 1.43318871
 B3 1.0909973
 B4 1.09072868
 B5 1.09287012
 B6 1.50793568
 B7 1.09469424
 B8 1.09673641
 B9 1.09609604
  )
symmetry c1
"""
#)

psi4.set_options({'basis': '3-21g',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'false',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10,
                  'diis': 1})

# CCSD Settings

#precision (0 for dp, 1 for mp)
precision = 1
print_amps = False

# RHF from Psi4
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

# Energy
if precision == 0:
    ccsd = ccEnergy_dp(mol, rhf_e, rhf_wfn, numpy_memory = 2)
elif precision == 1:
    ccsd = ccEnergy(mol, rhf_e, rhf_wfn, numpy_memory=2)
else:
    print("Please check the input of precision!")

ccsd_corr_e = ccsd.compute_energy(e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=50,
                       max_diis=8,
                       start_diis=100)
print(ccsd_corr_e, ccsd_corr_e + ccsd.SCF_E)
#psi4.compare_values(psi4.energy('CCSD'), ccsd_corr_e + ccsd.SCF_E, 6, 'CCSD Energy')
#6h2o, 3-21g
#psi4.compare_values(-454.4553648318574, ccsd_corr_e + ccsd.SCF_E, 6, 'CCSD Energy')
#uracil, 3-21g
#psi4.compare_values(-410.98611298, ccsd_corr_e + ccsd.SCF_E, 6, 'CCSD Energy')

#7h2o, 3-21g
#psi4.compare_values(-530.2299494434577, ccsd_corr_e + ccsd.SCF_E, 6, 'CCSD Energy')
#6-31g
#psi4.compare_values(-532.978862007108, ccsd_corr_e + ccsd.SCF_E, 6, 'CCSD Energy')
#631g*
#psi4.compare_values(-533.578062728424584, ccsd_corr_e + ccsd.SCF_E, 6, 'CCSD Energy')
#--------------------------------------------------------------------------------------

# Pseudo-energy
# Lambda
if precision == 0:
    cclambda = ccLambda_dp(ccsd)
elif precision == 1:
    cclambda = ccLambda(ccsd)
else:
    print("Please check the input of precision!")

pseudo_e = cclambda.compute_lambda(e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=100)
print(pseudo_e)
# 3H2O: pseudo_e = -0.3980603088172412
# 5H2O: pseudo_e = -0.666668388436126
# Benzene: pseudo_e = -0.558294031394414
# Uracil: pseudo_e = -0.8050780254090781
#-----------------------------------------------------------------------------------------

# Polarizability
# frequency of calculation
omega_nm = 589
# conver from nm into hartree
omega_nm = 589
omega = (pc.c * pc.h * 1e9) / (pc.hartree2J * omega_nm)

cart = ['X', 'Y', 'Z']
Mu = {}
ccpert = {}
polar_AB = {}

# Obtain AO Dipole Matrices From Mints
dipole_array = ccsd.mints.ao_dipole()

for i in range(0, 3):
    string = "MU_" + cart[i]

    # Transform dipole integrals from AO to MO basis
    Mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                           np.asarray(dipole_array[i]))

    # Initializing the perturbation classs corresponding to dipole perturabtion at the given omega
    if precision == 0:
        ccpert[string] = CCPert_dp(string, Mu[string], ccsd, cclambda, omega)
    elif precision == 1:
        ccpert[string] = CCPert(string, Mu[string], ccsd, cclambda, omega)
    else:
        print("Please check the input of precision!")

    # Solve X and Y amplitudes corresponding to dipole perturabtion at the given omega
    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert[string].solve('right', r_conv=1e-7)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert[string].solve('left', r_conv=1e-7)

# Please refer to eq. 94 of [Koch:1991:3333] for the general form of linear response functions.
# For electric dipole polarizabilities, A = mu[x/y/z] and B = mu[x/y/z],
# Ex. alpha_xy = <<mu_x;mu_y>>, where mu_x = x and mu_y = y

print("\nComputing <<Mu;Mu> tensor @ %d nm" % omega_nm)

for a in range(0, 3):
    str_a = "MU_" + cart[a]
    for b in range(0, 3):
        str_b = "MU_" + cart[b]
        # Computing the alpha tensor
        if precision == 0:
            polar_AB[3 * a + b] = CCLinresp_dp(cclambda, ccpert[str_a], ccpert[str_b]).linresp()
        elif precision == 1:
            polar_AB[3 * a + b] = CCLinresp(cclambda, ccpert[str_a], ccpert[str_b]).linresp_dp()
        else:
            print("Please check the input of precision!")

# Symmetrizing the tensor
for a in range(0, 3):
    for b in range(0, a + 1):
        ab = 3 * a + b
        ba = 3 * b + a
        if a != b:
            polar_AB[ab] = 0.5 * (polar_AB[ab] + polar_AB[ba])
            polar_AB[ba] = polar_AB[ab]

print(
    '\nCCSD Dipole Polarizability Tensor (Length Gauge) at omega = %8.6lf au, %d nm\n'
    % (omega, omega_nm))
print("\t\t%s\t             %s\t                  %s\n" % (cart[0], cart[1],
                                                           cart[2]))

for a in range(0, 3):
    print(" %s %20.10lf %20.10lf %20.10lf\n" %
          (cart[a], polar_AB[3 * a + 0], polar_AB[3 * a + 1],
           polar_AB[3 * a + 2]))

# Isotropic polarizability
trace = polar_AB[0] + polar_AB[4] + polar_AB[8]
Isotropic_polar = trace / 3.0

print(
    " Isotropic CCSD Dipole Polarizability @ %d nm (Length Gauge): %20.10lf a.u."
    % (omega_nm, Isotropic_polar))

# Comaprison with PSI4 (if you have near to latest version of psi4)
psi4.set_options({'d_convergence': 1e-10})
psi4.set_options({'e_convergence': 1e-10})
psi4.set_options({'r_convergence': 1e-10})
psi4.set_options({'omega': [589, 'nm']})
psi4.properties('ccsd', properties=['polarizability'])
psi4.compare_values(Isotropic_polar, psi4.variable("CCSD DIPOLE POLARIZABILITY @ 589NM"),  6, "CCSD Isotropic Dipole Polarizability @ 589 nm (Length Gauge)") #TEST


#psi4.compare_values(
#    Isotropic_polar, 8.5698293248, 6,
#    "CCSD Isotropic Dipole Polarizability @ 589 nm (Length Gauge)")  # TEST
