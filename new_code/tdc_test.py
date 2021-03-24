# Input
#(Final version with dp, mp and sp 02/25/2022)
# settings (psi4, mol, field, propagation, plots)
# ground state CCSD
# propagation
# spectral property

import psi4
from psi4 import constants as pc
import numpy as np
import molecules
from ccenergy_gen import ccEnergy
from ccenergy_sp import ccEnergy_sp

np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

## Setup and parameters

# Psi4 setup
psi4.set_memory('2 GB')
psi4.core.set_output_file('output.dat', False)
memory = 4

psi4.set_options({'basis': 'cc-pVDZ',
                  'scf_type': 'pk',
                  'mp2_type': 'conv',
                  'freeze_core': 'true',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10,
                  'r_convergence': 1e-10,
                  'diis': 1})

# CCSD Settings

mol = psi4.geometry(molecules.mollist["(H2O)_2"])

# RHF from Psi4
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

max_diis=8
start_diis=1
maxiter=75

e_conv=1e-10
r_conv=1e-10
precision="double"
ccsd_dp = ccEnergy(mol, rhf_wfn, precision, memory)
e_dp = ccsd_dp.compute_energy(e_conv,r_conv,maxiter,max_diis,start_diis)

e_conv=1e-10
r_conv=1e-7
precision="mixed"
ccsd_mp = ccEnergy(mol, rhf_wfn, precision, memory)
e_mp = ccsd_mp.compute_energy(e_conv,r_conv,maxiter,max_diis,start_diis)

e_conv=1e-10
r_conv=1e-7
precision="single"
ccsd_sp = ccEnergy(mol, rhf_wfn, precision, memory)
e_sp = ccsd_sp.compute_energy(e_conv,r_conv,maxiter,max_diis,start_diis)

#psi4.energy('CCSD')

#e_conv=1e-10
#r_conv=1e-7
#max_diis=8
#ccsd_sp_old = ccEnergy_sp(mol, rhf_e, rhf_wfn, memory)
#e_sp_old = ccsd_sp_old.compute_energy(e_conv,r_conv,maxiter,max_diis,start_diis)

print("DP      = %20.14f" % e_dp)
print("MP      = %20.14f  %4.2e" % (e_mp, abs(e_dp-e_mp)))
print("SP      = %20.14f  %4.2e" % (e_sp, abs(e_dp-e_sp)))
#print("SP(old) = %20.14f  %4.2e" % (e_sp_old, abs(e_dp-e_sp_old)))
