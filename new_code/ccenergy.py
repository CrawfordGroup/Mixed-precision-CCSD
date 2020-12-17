# ccenergy
# ground state energy
# initial T-amplitudes for the real-time propagation
# ftns for the CCSD calculations during the propagation

import psi4
import time
import numpy as np
from utils import ndot
from utils import helper_diis
from opt_einsum import contract
np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

class ccEnergy(object):
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

        self.maxiter = 20

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
        #self.P = self.get_P()

        # Occupied and Virtual orbital energies for the denominators
        Focc = np.diag(self.F)[self.slice_o]
        Fvir = np.diag(self.F)[self.slice_v]

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

    # get_F can be used for both sp and dp
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
        tau = t2.copy()
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        tau += tmp
        return tau

# Modify the intermediates and the t-amp eqns to mixed-precision
    # Build Eqn 3:
    def build_Fae(self, F, Fsp, t1_sp, ttau):
        Fae = self.get_F(F, 'vv').copy()
        tmp = ndot('me,ma->ae', self.get_F(Fsp, 'ov'), t1_sp, prefactor=0.5)
        Fae -= np.float64(tmp)
        # the next tmp are in the same perturbative order
        tmp_mo = 2.0 * self.get_MO_sp('ovvv') - 1.0 * self.get_MO_sp('ovvv').swapaxes(2, 3)
        tmp = ndot('mf,mafe->ae', t1_sp, tmp_mo, prefactor=1.0)
        #tmp = ndot('mf,mafe->ae', t1_sp, self.get_MO_sp('ovvv'), prefactor=2.0)
        #tmp += ndot('mf,maef->ae', t1_sp, self.get_MO_sp('ovvv'), prefactor=-1.0)
        Fae += np.float64(tmp)
        tmp_mo = 2.0 * self.get_MO_sp('oovv') - 1.0 * self.get_MO_sp('oovv').swapaxes(2, 3)
        tmp = ndot('mnaf,mnef->ae', np.float32(ttau), tmp_mo, prefactor=1.0)
        #tmp = ndot('mnaf,mnef->ae', np.float32(ttau), self.get_MO_sp('oovv'), prefactor=2.0)
        #tmp += ndot('mnaf,mnfe->ae', np.float32(ttau), self.get_MO_sp('oovv'), prefactor=-1.0)
        Fae -= np.float64(tmp)
        return Fae

    # Build Eqn 4:
    def build_Fmi(self, F, Fsp, t1_sp, ttau):
        Fmi = self.get_F(F, 'oo').copy()
        tmp = ndot('ie,me->mi', t1_sp, self.get_F(Fsp, 'ov'), prefactor=0.5)
        Fmi += np.float64(tmp)
        tmp = ndot('ne,mnie->mi', t1_sp, self.get_MO_sp('ooov'), prefactor=2.0)
        tmp += ndot('ne,mnei->mi', t1_sp, self.get_MO_sp('oovo'), prefactor=-1.0)
        Fmi += np.float64(tmp)
        tmp = ndot('inef,mnef->mi', np.float32(ttau), self.get_MO_sp('oovv'), prefactor=2.0)
        tmp += ndot('inef,mnfe->mi', np.float32(ttau), self.get_MO_sp('oovv'), prefactor=-1.0)
        Fmi += np.float64(tmp)
        return Fmi

        # Build Eqn 5:

    def build_Fme(self, F, t1_sp):
        Fme = self.get_F(F, 'ov').copy()
        tmp_mo = 2.0 * self.get_MO_sp('oovv') - 1.0 * self.get_MO_sp('oovv').swapaxes(2, 3)
        tmp = ndot('nf,mnef->me', t1_sp, tmp_mo, prefactor=2.0)
        #tmp = ndot('nf,mnef->me', t1_sp, self.get_MO_sp('oovv'), prefactor=2.0)
        #tmp += ndot('nf,mnfe->me', t1_sp, self.get_MO_sp('oovv'), prefactor=-1.0)
        Fme += np.float64(tmp)
        return Fme

    # Build Eqn 6:
    def build_Wmnij(self, t1_sp, tau):
        Wmnij = self.get_MO('oooo').copy()
        tmp = ndot('je,mnie->mnij', t1_sp, self.get_MO_sp('ooov'))
        tmp += ndot('ie,mnej->mnij', t1_sp, self.get_MO_sp('oovo'))
        Wmnij += np.float64(tmp)
        # prefactor of 1 instead of 0.5 below to fold the last term of
        # 0.5 * tau_ijef Wabef in Wmnij contraction: 0.5 * tau_mnab Wmnij_mnij
        tmp = ndot('ijef,mnef->mnij', np.float32(tau), self.get_MO_sp('oovv'), prefactor=1.0)
        Wmnij += np.float64(tmp)
        return Wmnij

    # Build Eqn 8:
    def build_Wmbej(self, t1, t2, t1_sp, t2_sp):
        Wmbej = self.get_MO('ovvo').copy()
        Wmbej += np.float64(ndot('jf,mbef->mbej', t1_sp, self.get_MO_sp('ovvv')))
        Wmbej -= np.float64(ndot('nb,mnej->mbej', t1_sp, self.get_MO_sp('oovo')))
        tmp_t = (0.5 * t2)
        tmp_t += np.einsum('jf,nb->jnfb', t1, t1)
        tmp = ndot('jnfb,mnef->mbej', np.float32(tmp_t), self.get_MO_sp('oovv'))
        tmp -= ndot('njfb,mnef->mbej', t2_sp, self.get_MO_sp('oovv'), prefactor=1.0)
        tmp -= ndot('njfb,mnfe->mbej', t2_sp, self.get_MO_sp('oovv'), prefactor=-0.5)
        Wmbej -= np.float64(tmp)
        return Wmbej

    # This intermediate appaears in the spin factorization of Wmbej terms.
    def build_Wmbje(self, t1, t2, t1_sp):
        Wmbje = -1.0 * (self.get_MO('ovov').copy())
        Wmbje -= np.float64(ndot('jf,mbfe->mbje', t1_sp, self.get_MO_sp('ovvv')))
        Wmbje += np.float64(ndot('nb,mnje->mbje', t1_sp, self.get_MO_sp('ooov')))
        tmp_t = (0.5 * t2)
        tmp_t += np.einsum('jf,nb->jnfb', t1, t1)
        Wmbje += np.float64(ndot('jnfb,mnfe->mbje', np.float32(tmp_t), self.get_MO_sp('oovv')))
        return Wmbje

    # This intermediate is required to build second term of 0.5 * tau_ijef * Wabef,
    # as explicit construction of Wabef is avoided here.
    def build_Zmbij(self, tau):
        Zmbij = 0
        Zmbij += np.float64(ndot('mbef,ijef->mbij', self.get_MO_sp('ovvv'), np.float32(tau)))
        return Zmbij

    # Compute the RHS of i*(d(t_amp)/dt = <fi_exc|H(t)T|fi(0)>
    # (-i)*(Residule of t1(t2) - t1 * Dia (t2 * Dijab))

    def r_T1(self, F, Fsp, t1_sp, t2_sp, ttau):
        # Single-precision intermediates
        Fae = np.float32(self.build_Fae(F, Fsp, t1_sp, ttau))
        Fme = np.float32(self.build_Fme(F, t1_sp))
        Fmi = np.float32(self.build_Fmi(F, Fsp, t1_sp, ttau))
        #### Build residual of T1 equations by spin adaption of  Eqn 1:
        r_T1 = self.get_F(F, 'ov').copy()
        r_T1 += np.float64(ndot('ie,ae->ia', t1_sp, Fae))
        r_T1 -= np.float64(ndot('ma,mi->ia', t1_sp, Fmi))
        tmp = ndot('imae,me->ia', t2_sp, Fme, prefactor=2.0)
        tmp += ndot('imea,me->ia', t2_sp, Fme, prefactor=-1.0)
        r_T1 += np.float64(tmp)
        tmp = ndot('nf,nafi->ia', t1_sp, self.get_MO_sp('ovvo'), prefactor=2.0)
        tmp += ndot('nf,naif->ia', t1_sp, self.get_MO_sp('ovov'), prefactor=-1.0)
        r_T1 += np.float64(tmp)
        tmp = ndot('mief,maef->ia', t2_sp, self.get_MO_sp('ovvv'), prefactor=2.0)
        tmp += ndot('mife,maef->ia', t2_sp, self.get_MO_sp('ovvv'), prefactor=-1.0)
        r_T1 += np.float64(tmp)
        tmp = ndot('mnae,nmei->ia', t2_sp, self.get_MO_sp('oovo'), prefactor=2.0)
        tmp += ndot('mnae,nmie->ia', t2_sp, self.get_MO_sp('ooov'), prefactor=-1.0)
        r_T1 -= np.float64(tmp)

        # return (r_T1 - t1 / Dia) * (-1.0)
        return r_T1

    def r_T2(self, F, Fsp, t1, t2, t1_sp, t2_sp, tau, ttau):
        Fae = np.float32(self.build_Fae(F, Fsp, t1_sp, ttau))
        Fme = np.float32(self.build_Fme(F, t1_sp))
        Fmi = np.float32(self.build_Fmi(F, Fsp, t1_sp, ttau))
        # <ij||ab> ->  <ij|ab>
        #   spin   ->  spin-adapted (<alpha beta| alpha beta>)
        r_T2 = self.get_MO('oovv').copy()

        # Conventions used:
        #   P(ab) f(a,b) = f(a,b) - f(b,a)
        #   P(ij) f(i,j) = f(i,j) - f(j,i)
        #   P^(ab)_(ij) f(a,b,i,j) = f(a,b,i,j) + f(b,a,j,i)

        # P(ab) {t_ijae Fae_be}  ->  P^(ab)_(ij) {t_ijae Fae_be}
        tmp = ndot('ijae,be->ijab', t2_sp, Fae)
        tmp += tmp.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 += np.float64(tmp)

        # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
        # P(ab) {-0.5 * t_ijae t_mb Fme_me} -> P^(ab)_(ij) {-0.5 * t_ijae t_mb Fme_me}
        tmp = ndot('mb,me->be', t1_sp, Fme)
        first = ndot('ijae,be->ijab', t2_sp, tmp, prefactor=0.5)
        first += first.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 -= np.float64(first)

        # P(ij) {-t_imab Fmi_mj}  ->  P^(ab)_(ij) {-t_imab Fmi_mj}
        tmp = ndot('imab,mj->ijab', t2_sp, Fmi, prefactor=1.0)
        tmp += tmp.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 -= np.float64(tmp)

        # P(ij) {-0.5 * t_imab t_je Fme_me}  -> P^(ab)_(ij) {-0.5 * t_imab t_je Fme_me}
        tmp = ndot('je,me->jm', t1_sp, Fme)
        first = ndot('imab,jm->ijab', t2_sp, tmp, prefactor=0.5)
        first += first.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 -= np.float64(first)

        # Build TEI Intermediates
        tmp_tau_sp = np.float32(self.build_tau(t1, t2))
        Wmnij = np.float32(self.build_Wmnij(t1_sp, tau))
        Wmbej = np.float32(self.build_Wmbej(t1, t2, t1_sp, t2_sp))
        Wmbje = np.float32(self.build_Wmbje(t1, t2, t1_sp))
        Zmbij = np.float32(self.build_Zmbij(tau))

        # 0.5 * tau_mnab Wmnij_mnij  -> tau_mnab Wmnij_mnij
        # This also includes the last term in 0.5 * tau_ijef Wabef
        # as Wmnij is modified to include this contribution.
        r_T2 += np.float64(ndot('mnab,mnij->ijab', tmp_tau_sp, Wmnij, prefactor=1.0))

        # Wabef used in eqn 2 of reference 1 is very expensive to build and store, so we have
        # broken down the term , 0.5 * tau_ijef * Wabef (eqn. 7) into different components
        # The last term in the contraction 0.5 * tau_ijef * Wabef is already accounted
        # for in the contraction just above.

        # First term: 0.5 * tau_ijef <ab||ef> -> tau_ijef <ab|ef>
        r_T2 += np.float64(ndot('ijef,abef->ijab', tmp_tau_sp, self.get_MO_sp('vvvv'), prefactor=1.0))

        # Second term: 0.5 * tau_ijef (-P(ab) t_mb <am||ef>)  -> -P^(ab)_(ij) {t_ma * Zmbij_mbij}
        # where Zmbij_mbij = <mb|ef> * tau_ijef
        tmp = ndot('ma,mbij->ijab', t1_sp, Zmbij)
        tmp += tmp.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 -= np.float64(tmp)

        # P(ij)P(ab) t_imae Wmbej -> Broken down into three terms below
        # First term: P^(ab)_(ij) {(t_imae - t_imea)* Wmbej_mbej}
        tmp1 = ndot('imae,mbej->ijab', t2_sp, Wmbej, prefactor=1.0)
        tmp1 += ndot('imea,mbej->ijab', t2_sp, Wmbej, prefactor=-1.0)
        tmp1 += tmp1.swapaxes(0, 1).swapaxes(2, 3)

        # Second term: P^(ab)_(ij) t_imae * (Wmbej_mbej + Wmbje_mbje)
        tmp2 = ndot('imae,mbej->ijab', t2_sp, Wmbej, prefactor=1.0)
        tmp2 += ndot('imae,mbje->ijab', t2_sp, Wmbje, prefactor=1.0)
        tmp2 += tmp2.swapaxes(0, 1).swapaxes(2, 3)

        # Third term: P^(ab)_(ij) t_mjae * Wmbje_mbie
        tmp3 = ndot('mjae,mbie->ijab', t2_sp, Wmbje, prefactor=1.0)
        tmp3 += tmp3.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 += np.float64(tmp1 + tmp2 + tmp3)

        # -P(ij)P(ab) {-t_ie * t_ma * <mb||ej>} -> P^(ab)_(ij) {-t_ie * t_ma * <mb|ej>
        #                                                      + t_ie * t_mb * <ma|je>}
        tmp = ndot('ie,ma->imea', t1_sp, t1_sp)
        tmp1 = ndot('imea,mbej->ijab', tmp, self.get_MO_sp('ovvo'))
        tmp1 += tmp1.swapaxes(0, 1).swapaxes(2, 3)

        tmp = ndot('ie,mb->imeb', t1_sp, t1_sp)
        tmp2 = ndot('imeb,maje->ijab', tmp, self.get_MO_sp('ovov'))
        tmp2 += tmp2.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 -= np.float64(tmp1 + tmp2)

        # P(ij) {t_ie <ab||ej>} -> P^(ab)_(ij) {t_ie <ab|ej>}
        tmp = ndot('ie,abej->ijab', t1_sp, self.get_MO_sp('vvvo'), prefactor=1.0)
        tmp += tmp.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 += np.float64(tmp)

        # P(ab) {-t_ma <mb||ij>} -> P^(ab)_(ij) {-t_ma <mb|ij>}
        tmp = ndot('ma,mbij->ijab', t1_sp, self.get_MO_sp('ovoo'), prefactor=1.0)
        tmp += tmp.swapaxes(0, 1).swapaxes(2, 3)
        r_T2 -= np.float64(tmp)

        return r_T2
        # return (r_T2 - t2 / Dijab) * (-1.0)

    # Compute the ground state t-amplitudes for the start point (t=0)
    def update(self):
        ### Update T1 and T2 amplitudes
        tau = self.build_tau(self.t1, self.t2)
        ttau = self.build_tilde_tau(self.t1, self.t2)
        r1 = self.r_T1(self.F, self.Fsp, self.t1_sp, self.t2_sp, ttau)
        r2 = self.r_T2(self.F, self.Fsp, self.t1, self.t2, self.t1_sp, self.t2_sp, tau, ttau)
        self.t1 += r1 / self.Dia
        self.t2 += r2 / self.Dijab
        self.t1_sp = np.float32(self.t1)
        self.t2_sp = np.float32(self.t2)
        #print(np.linalg.norm(r1), np.linalg.norm(r2))
        rms = np.einsum('ia,ia->', r1 / self.Dia, r1 / self.Dia)
        rms += np.einsum('ijab,ijab->', r2 / self.Dijab, r2 / self.Dijab)

        return np.sqrt(rms)

    # Compute energy, needed in the propagation
    def compute_corr_energy(self, F, t1, t2):
        CCSDcorr_E = 2.0 * np.einsum('ia,ia->', self.get_F(F, 'ov'), t1)
        tmp_tau = self.build_tau(t1, t2)
        CCSDcorr_E += 2.0 * np.einsum('ijab,ijab->', tmp_tau, self.get_MO('oovv'))
        CCSDcorr_E -= 1.0 * np.einsum('ijab,ijba->', tmp_tau, self.get_MO('oovv'))

        return CCSDcorr_E

    def compute_energy(self,
                       e_conv=1e-6,
                       r_conv=1e-4,
                       maxiter=50,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        ccsd_tstart = time.time()
        # Compute MP2 energy
        CCSDcorr_E_old = self.compute_corr_energy(self.F, self.t1, self.t2)
        print("CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   MP2" % (0, np.real(CCSDcorr_E_old), np.real(-CCSDcorr_E_old)))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterations
        for CCSD_iter in range(1, maxiter + 1):

            rms = self.update()

            # Compute CCSD correlation energy
            CCSDcorr_E = self.compute_corr_energy(self.F, self.t1, self.t2)

            # Print CCSD iteration information
            print('CCSD Iteration %3d: CCSD correlation = %.15f   dE = % .5E   DIIS = %d' % (
            CCSD_iter, CCSDcorr_E, CCSDcorr_E - CCSDcorr_E_old, diis_object.diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv and rms < r_conv):
                print('\nCCSD has converged in %.3f seconds!' % (time.time() - ccsd_tstart))
                return CCSDcorr_E
                # return t1_init

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if CCSD_iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)
