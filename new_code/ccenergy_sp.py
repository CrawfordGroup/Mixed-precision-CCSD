# ccenergy in double-precision
# ground state energy

import psi4
import time
import numpy as np
from utils import ndot
from utils import helper_diis
np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

class ccEnergy_sp(object):
    def __init__(self, mol, rhf_e, rhf_wfn, numpy_memory = 2):
        print("\nInitalizing CCSD object...\n")
        time_init = time.time()
        # RHF from Psi4
        self.rhf_e = rhf_e
        self.wfn = rhf_wfn

        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.H = np.asarray(self.mints.ao_kinetic()) + np.asarray(self.mints.ao_potential())

        self.ndocc = self.wfn.doccpi()[0]
        self.nmo = self.wfn.nmo()
        self.nmo = self.H.shape[0]
        self.memory = numpy_memory
        self.C = self.wfn.Ca()
        self.npC = np.asarray(self.C)

        self.SCF_E = self.wfn.energy()

        # Update H, transform to MO basis
        self.H = np.einsum('uj,vi,uv', self.npC, self.npC, self.H)

        print('Starting AO ->  MO transformation...')

        ERI_Size = self.nmo * 128.e-9
        memory_footprint = ERI_Size * 5
        if memory_footprint > self.memory:
            psi4.clean()
            raise Exception(
                "Estimated memory utilization (%4.2f GB) exceeds numpy_memory \
                            limit of %4.2f GB." % (memory_footprint, memory))

        # Integral generation from Psi4's MintsHelper
        self.MO = np.asarray(self.mints.mo_eri(self.C, self.C, self.C, self.C))
        # Physicist notation
        self.MO = self.MO.swapaxes(1, 2)
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." %
              (ERI_Size, self.nmo))
        # single-precision MO integrals
        self.MOsp = np.float32(self.MO)

        # Update nocc and nvirt
        self.nocc = self.ndocc
        self.nvirt = self.nmo - self.nocc

        # Make slices
        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }

        # Compute Fock matrix
        self.F = self.H + 2.0 * np.einsum('pmqm->pq', self.MO[:, self.slice_o, :, self.slice_o])
        self.F -= np.einsum('pmmq->pq', self.MO[:, self.slice_o, self.slice_o, :])
        # Single-precision F
        self.Fsp = np.float32(self.F)

        # Compute AO density
        self.P = self.get_P()

        # Occupied and Virtual orbital energies for the denominators
        Focc = np.diag(self.Fsp)[self.slice_o]
        Fvir = np.diag(self.Fsp)[self.slice_v]

        # Denominator
        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir

        # Construct initial guess of t1, t2 (t1, t2 at t=0)
        print('Building initial guess...')
        # t^a_i
        self.t1 = np.zeros((self.nocc, self.nvirt))
        # t^{ab}_{ij}
        self.t2 = self.MO[self.slice_o, self.slice_o, self.slice_v, self.slice_v] / self.Dijab
        # single-precision t1, t2
        self.t1_sp = np.float32(self.t1)
        self.t2_sp = np.float32(self.t2)


        print('\n..initialized CCSD in %.3f seconds.\n' % (time.time() - time_init))

    def get_MO(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                  self.slice_dict[string[2]], self.slice_dict[string[3]]]
    def get_MO_sp(self, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MOsp[self.slice_dict[string[0]], self.slice_dict[string[1]],
                  self.slice_dict[string[2]], self.slice_dict[string[3]]]

    # Can be used for sp
    def get_F(self, F, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    # build AO density (for HF dipole)
    def get_P(self):
        o = self.slice_o
        C = self.npC
        P = ndot('ui,vi->uv', C[:,o], C[:,o], prefactor=2)
        return P

    # Equations from Reference 1 (Stanton's paper)

    # Bulid Eqn 9:
    def build_tilde_tau(self, t1, t2):
        ttau = t2.copy()
        tmp = 0.5 * np.einsum('ia,jb->ijab', t1, t1)
        # print("tmp", t1, ttau, tmp)
        ttau += tmp
        return ttau

    # Build Eqn 10:
    def build_tau(self, t1, t2):
        ttau = t2.copy()
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        ttau += tmp
        return ttau

    # Build Eqn 3:
    def build_Fae(self, F, t1, t2):
        Fae = self.get_F(F, 'vv').copy()
        Fae -= ndot('me,ma->ae', self.get_F(F, 'ov'), t1, prefactor=0.5)
        Fae += ndot('mf,mafe->ae', t1, self.get_MO_sp('ovvv'), prefactor=2.0)
        Fae += ndot('mf,maef->ae', t1, self.get_MO_sp('ovvv'), prefactor=-1.0)
        Fae -= ndot('mnaf,mnef->ae', self.build_tilde_tau(t1, t2), self.get_MO_sp('oovv'), prefactor=2.0)
        Fae -= ndot('mnaf,mnfe->ae', self.build_tilde_tau(t1, t2), self.get_MO_sp('oovv'), prefactor=-1.0)
        return Fae

    # Build Eqn 4:
    def build_Fmi(self, F, t1, t2):
        Fmi = self.get_F(F, 'oo').copy()
        Fmi += ndot('ie,me->mi', t1, self.get_F(F, 'ov'), prefactor=0.5)
        Fmi += ndot('ne,mnie->mi', t1, self.get_MO_sp('ooov'), prefactor=2.0)
        Fmi += ndot('ne,mnei->mi', t1, self.get_MO_sp('oovo'), prefactor=-1.0)
        Fmi += ndot('inef,mnef->mi', self.build_tilde_tau(t1, t2), self.get_MO_sp('oovv'), prefactor=2.0)
        Fmi += ndot('inef,mnfe->mi', self.build_tilde_tau(t1, t2), self.get_MO_sp('oovv'), prefactor=-1.0)
        return Fmi

        # Build Eqn 5:

    def build_Fme(self, F, t1):
        Fme = self.get_F(F, 'ov').copy()
        Fme += ndot('nf,mnef->me', t1, self.get_MO_sp('oovv'), prefactor=2.0)
        Fme += ndot('nf,mnfe->me', t1, self.get_MO_sp('oovv'), prefactor=-1.0)
        return Fme

    # Build Eqn 6:
    def build_Wmnij(self, t1, t2):
        Wmnij = self.get_MO_sp('oooo').copy()
        Wmnij += ndot('je,mnie->mnij', t1, self.get_MO_sp('ooov'))
        Wmnij += ndot('ie,mnej->mnij', t1, self.get_MO_sp('oovo'))
        # prefactor of 1 instead of 0.5 below to fold the last term of
        # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
        Wmnij += ndot('ijef,mnef->mnij', self.build_tau(t1, t2), self.get_MO_sp('oovv'), prefactor=1.0)
        return Wmnij

    # Build Eqn 8:
    def build_Wmbej(self, t1, t2):
        Wmbej = self.get_MO_sp('ovvo').copy()
        Wmbej += ndot('jf,mbef->mbej', t1, self.get_MO_sp('ovvv'))
        Wmbej -= ndot('nb,mnej->mbej', t1, self.get_MO_sp('oovo'))
        tmp = (0.5 * t2)
        tmp += np.einsum('jf,nb->jnfb', t1, t1)
        Wmbej -= ndot('jnfb,mnef->mbej', tmp, self.get_MO_sp('oovv'))
        Wmbej += ndot('njfb,mnef->mbej', t2, self.get_MO_sp('oovv'), prefactor=1.0)
        Wmbej += ndot('njfb,mnfe->mbej', t2, self.get_MO_sp('oovv'), prefactor=-0.5)
        return Wmbej

    # This intermediate appaears in the spin factorization of Wmbej terms.
    def build_Wmbje(self, t1, t2):
        Wmbje = -1.0 * (self.get_MO_sp('ovov').copy())
        Wmbje -= ndot('jf,mbfe->mbje', t1, self.get_MO_sp('ovvv'))
        Wmbje += ndot('nb,mnje->mbje', t1, self.get_MO_sp('ooov'))
        tmp = (0.5 * t2)
        tmp += np.einsum('jf,nb->jnfb', t1, t1)
        Wmbje += ndot('jnfb,mnfe->mbje', tmp, self.get_MO_sp('oovv'))
        return Wmbje

    # This intermediate is required to build second term of 0.5 * tau_ijef * Wabef,
    # as explicit construction of Wabef is avoided here.
    def build_Zmbij(self, t1, t2):
        Zmbij = 0
        Zmbij += ndot('mbef,ijef->mbij', self.get_MO_sp('ovvv'), self.build_tau(t1, t2))
        return Zmbij

    # Compute the RHS of i*(d(t_amp)/dt = <fi_exc|H(t)T|fi(0)>
    # (-i)*(Residule of t1(t2) - t1 * Dia (t2 * Dijab))

    def r_T1(self, F, t1, t2):
        Fae = self.build_Fae(F, t1, t2)
        Fme = self.build_Fme(F, t1)
        Fmi = self.build_Fmi(F, t1, t2)
        #### Build residual of T1 equations by spin adaption of  Eqn 1:
        r_T1 = self.get_F(F, 'ov').copy()
        r_T1 += ndot('ie,ae->ia', t1, Fae)
        r_T1 -= ndot('ma,mi->ia', t1, Fmi)
        r_T1 += ndot('imae,me->ia', t2, Fme, prefactor=2.0)
        r_T1 += ndot('imea,me->ia', t2, Fme, prefactor=-1.0)
        r_T1 += ndot('nf,nafi->ia', t1, self.get_MO_sp('ovvo'), prefactor=2.0)
        r_T1 += ndot('nf,naif->ia', t1, self.get_MO_sp('ovov'), prefactor=-1.0)
        r_T1 += ndot('mief,maef->ia', t2, self.get_MO_sp('ovvv'), prefactor=2.0)
        r_T1 += ndot('mife,maef->ia', t2, self.get_MO_sp('ovvv'), prefactor=-1.0)
        r_T1 -= ndot('mnae,nmei->ia', t2, self.get_MO_sp('oovo'), prefactor=2.0)
        r_T1 -= ndot('mnae,nmie->ia', t2, self.get_MO_sp('ooov'), prefactor=-1.0)

        return r_T1

    def r_T2(self, F, t1, t2):
        Fae = self.build_Fae(F, t1, t2)
        Fme = self.build_Fme(F, t1)
        Fmi = self.build_Fmi(F, t1, t2)
        # <ij||ab> ->  <ij|ab>
        #   spin   ->  spin-adapted (<alpha beta| alpha beta>)
        r_T2 = self.get_MO_sp('oovv').copy()

        # Conventions used:
        #   P(ab) f(a,b) = f(a,b) - f(b,a)
        #   P(ij) f(i,j) = f(i,j) - f(j,i)
        #   P^(ab)_(ij) f(a,b,i,j) = f(a,b,i,j) + f(b,a,j,i)

        # P(ab) {t_ijae Fae_be}  ->  P^(ab)_(ij) {t_ijae Fae_be}
        tmp = ndot('ijae,be->ijab', t2, Fae)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
        tmp = ndot('mb,me->be', t1, Fme)
        first = ndot('ijae,be->ijab', t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
        tmp = ndot('imab,mj->ijab', t2, Fmi, prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
        tmp = ndot('je,me->jm', t1, Fme)
        first = ndot('imab,jm->ijab', t2, tmp, prefactor=0.5)
        r_T2 -= first
        r_T2 -= first.swapaxes(0, 1).swapaxes(2, 3)

        # Build TEI Intermediates
        tmp_tau = self.build_tau(t1, t2)
        Wmnij = self.build_Wmnij(t1, t2)
        Wmbej = self.build_Wmbej(t1, t2)
        Wmbje = self.build_Wmbje(t1, t2)
        Zmbij = self.build_Zmbij(t1, t2)

        # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
        # This also includes the last term in 0.5 * tau_ijef Wabef
        # as Wmnij is modified to include this contribution.
        r_T2 += ndot('mnab,mnij->ijab', tmp_tau, Wmnij, prefactor=1.0)

        # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
        # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
        # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
        # for in the contraction just above.

        # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
        r_T2 += ndot('ijef,abef->ijab', tmp_tau, self.get_MO_sp('vvvv'), prefactor=1.0)

        # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
        # where Zmbij_mbij = <mb|ef> * tau_ijef
        tmp = ndot('ma,mbij->ijab', t1, Zmbij)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
        # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
        tmp = ndot('imae,mbej->ijab', t2, Wmbej, prefactor=1.0)
        tmp += ndot('imea,mbej->ijab', t2, Wmbej, prefactor=-1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
        tmp = ndot('imae,mbej->ijab', t2, Wmbej, prefactor=1.0)
        tmp += ndot('imae,mbje->ijab', t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
        tmp = ndot('mjae,mbie->ijab', t2, Wmbje, prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
        #                                                      + t_ie * t_mb * <ma|je>}
        tmp = ndot('ie,ma->imea', t1, t1)
        tmp1 = ndot('imea,mbej->ijab', tmp, self.get_MO_sp('ovvo'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)
        tmp = ndot('ie,mb->imeb', t1, t1)
        tmp1 = ndot('imeb,maje->ijab', tmp, self.get_MO_sp('ovov'))
        r_T2 -= tmp1
        r_T2 -= tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
        tmp = ndot('ie,abej->ijab', t1, self.get_MO_sp('vvvo'), prefactor=1.0)
        r_T2 += tmp
        r_T2 += tmp.swapaxes(0, 1).swapaxes(2, 3)

        # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
        tmp = ndot('ma,mbij->ijab', t1, self.get_MO_sp('ovoo'), prefactor=1.0)
        r_T2 -= tmp
        r_T2 -= tmp.swapaxes(0, 1).swapaxes(2, 3)

        return r_T2


    # Compute the ground state t-amplitudes for the start point (t=0)
    def update(self):
        ### Update T1 and T2 amplitudes
        r1 = self.r_T1(self.Fsp, self.t1_sp, self.t2_sp)
        r2 = self.r_T2(self.Fsp, self.t1_sp, self.t2_sp)

        self.t1_sp += r1 / self.Dia
        self.t2_sp += r2 / self.Dijab

        rms = np.einsum('ia,ia->', r1 / self.Dia, r1 / self.Dia)
        rms += np.einsum('ijab,ijab->', r2 / self.Dijab, r2 / self.Dijab)

        return np.sqrt(rms)

    def compute_corr_energy(self, F, t1, t2):
        CCSDcorr_E = 2.0 * np.einsum('ia,ia->', self.get_F(F, 'ov'), t1)
        tmp_tau = self.build_tau(t1, t2)
        CCSDcorr_E += 2.0 * np.einsum('ijab,ijab->', tmp_tau, self.get_MO_sp('oovv'))
        CCSDcorr_E -= 1.0 * np.einsum('ijab,ijba->', tmp_tau, self.get_MO_sp('oovv'))

        return CCSDcorr_E

    def compute_energy(self,
                       e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        ccsd_tstart = time.time()
        # Compute MP2 energy (dp)
        CCSDcorr_E_old = self.compute_corr_energy(self.F, self.t1, self.t2)
        print("CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   MP2" % (0, CCSDcorr_E_old, -CCSDcorr_E_old))
        CCSDcorr_E_old = np.float32(CCSDcorr_E_old)
        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterations
        for CCSD_iter in range(1, maxiter + 1):

            rms = self.update()

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy(self.Fsp, self.t1_sp, self.t2_sp)

            # Print CCSD iteration information
            print('CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   DIIS = %d' % (
            CCSD_iter, CCSDcorr_E, CCSDcorr_E - CCSDcorr_E_old, diis_object.diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv and rms < r_conv):
                self.t1 = np.float64(self.t1_sp)
                self.t2 = np.float64(self.t2_sp)
                print('\nCCSD has converged in %.3f seconds!' % (time.time() - ccsd_tstart))
                return CCSDcorr_E
                # return t1_init

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E
            self.t1 = np.float64(self.t1_sp)
            self.t2 = np.float64(self.t2_sp)

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if CCSD_iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)
