import psi4
import time
import numpy as np
from utils import ndot
from utils import helper_diis
from utils import careful_sum
np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

class ccEnergy(object):

    def get_ERI(self, ERI, string):
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_ERI: string %s must have 4 elements.' % string)
        return ERI[self.slice_dict[string[0]], self.slice_dict[string[1]],
                  self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, F, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 4 elements.' % string)
        return F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def __init__(self, mol, scf_wfn, precision="double", memory=2):
        print("\nInitalizing CCSD object...\n")
        time_init = time.time()

        # RHF from Psi4
        self.wfn = scf_wfn
        self.SCF_E = self.wfn.energy()

        self.nfzc = self.wfn.frzcpi()[0]
        self.nocc = self.wfn.doccpi()[0] - self.nfzc # active docc
        self.nmo = self.wfn.nmo() # all MOs/AOs
        self.nvirt = self.nmo - self.nocc - self.nfzc # active virtual

        self.C = self.wfn.Ca_subset("AO", "ACTIVE")
        self.npC = np.asarray(self.C)

        # transform Fock matrix to MO basis
        self.F = np.asarray(self.wfn.Fa())
        self.F = np.einsum('uj,vi,uv', self.npC, self.npC, self.F)

        print('Starting AO ->  MO transformation...')

        ERI_Size = (self.nmo**4)*8/(1024**3)
        print("Size of the ERI tensor is %4.2f GB, %d basis functions." % (ERI_Size, self.nmo))
        memory_footprint = ERI_Size * 5
        if memory_footprint > memory:
            psi4.core.clean()
            raise Exception(
                "Estimated memory utilization (%4.2f GB) exceeds requested memory \
                            limit of %4.2f GB." % (memory_footprint, memory))

        # Integral generation from Psi4's MintsHelper
        self.mints = psi4.core.MintsHelper(self.wfn.basisset())
        self.ERI = np.asarray(self.mints.mo_eri(self.C, self.C, self.C, self.C))

        # Physicist notation
        self.ERI = self.ERI.swapaxes(1, 2)
        self.L = 2.0 * self.ERI - self.ERI.swapaxes(2, 3)

        # Make slices
        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }

        if precision != "double" and precision != "single" and precision != "mixed":
            raise Exception("Unknown option for precision:", precision)
        self.precision = precision
        print("Using %s precision." % precision)

        if self.precision=="mixed": # keep a copy of the float64 ints
            self.F64 = self.F.copy()
            self.L64 = self.L.copy()
            self.ERI64 = self.get_ERI(self.ERI, 'oovv').copy()

        if self.precision == "single" or self.precision == "mixed":
            self.F = np.float32(self.F)
            self.L = np.float32(self.L)
            self.ERI = np.float32(self.ERI)

        # Occupied and Virtual orbital energies for the denominators
        if self.precision == "mixed": # keep denominators float64
            Focc = np.diag(self.F64)[self.slice_o]
            Fvir = np.diag(self.F64)[self.slice_v]
        else:
            Focc = np.diag(self.F)[self.slice_o]
            Fvir = np.diag(self.F)[self.slice_v]

        # Denominator
        self.Dia = Focc.reshape(-1, 1) - Fvir
        self.Dijab = Focc.reshape(-1, 1, 1, 1) + Focc.reshape(-1, 1, 1) - Fvir.reshape(-1, 1) - Fvir

        print('Building initial guess...')

        if self.precision == "single":
            self.t1 = np.zeros((self.nocc, self.nvirt), dtype='float32')
        else:
            self.t1 = np.zeros((self.nocc, self.nvirt)) # gives float64 by default

        if self.precision == "mixed": # default to float64 to start
            self.t2 = self.ERI64 / self.Dijab
        else:
            self.t2 = self.get_ERI(self.ERI, 'oovv') / self.Dijab

        print('\n..initialized CCSD in %.3f seconds.\n' % (time.time() - time_init))

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
    def build_Fae(self, F, L, t1, t2):
        terms = []
        terms.append(self.get_F(F, 'vv').copy())
        terms.append(ndot('me,ma->ae', self.get_F(F, 'ov'), t1, prefactor=-0.5))
        terms.append(ndot('mf,mafe->ae', t1, self.get_ERI(L, 'ovvv')))
        terms.append(ndot('mnaf,mnef->ae', self.build_tilde_tau(t1, t2), self.get_ERI(L, 'oovv'), prefactor=-1.0))

        Fae = careful_sum(terms, self.precision)

        return Fae

    # Build Eqn 4:
    def build_Fmi(self, F, L, t1, t2):
        terms = []
        terms.append(self.get_F(F, 'oo').copy())
        terms.append(ndot('ie,me->mi', t1, self.get_F(F, 'ov'), prefactor=0.5))
        terms.append(ndot('ne,mnie->mi', t1, self.get_ERI(L, 'ooov')))
        terms.append(ndot('inef,mnef->mi', self.build_tilde_tau(t1, t2), self.get_ERI(L, 'oovv')))

        Fmi = careful_sum(terms, self.precision)

        return Fmi

        # Build Eqn 5:

    def build_Fme(self, F, L, t1):
        terms = []
        terms.append(self.get_F(F, 'ov').copy())
        terms.append(ndot('nf,mnef->me', t1, self.get_ERI(L, 'oovv')))

        Fme = careful_sum(terms, self.precision)

        return Fme

    # Build Eqn 6:
    def build_Wmnij(self, ERI, t1, t2):
        terms = []
        terms.append(self.get_ERI(ERI, 'oooo').copy())
        terms.append(ndot('je,mnie->mnij', t1, self.get_ERI(ERI, 'ooov')))
        terms.append(ndot('ie,mnej->mnij', t1, self.get_ERI(ERI, 'oovo')))
        terms.append(ndot('ijef,mnef->mnij', self.build_tau(t1, t2), self.get_ERI(ERI, 'oovv')))

        Wmnij = careful_sum(terms, self.precision)

        return Wmnij

    # Build Eqn 8:
    def build_Wmbej(self, ERI, L, t1, t2):
        terms = []
        terms.append(self.get_ERI(ERI, 'ovvo').copy())
        terms.append(ndot('jf,mbef->mbej', t1, self.get_ERI(ERI, 'ovvv')))
        terms.append(ndot('nb,mnej->mbej', t1, self.get_ERI(ERI, 'oovo'), prefactor=-1.0))
        tmp = (0.5 * t2)
        tmp += np.einsum('jf,nb->jnfb', t1, t1)
        terms.append(ndot('jnfb,mnef->mbej', tmp, self.get_ERI(ERI, 'oovv'), prefactor=-1.0))
        terms.append(ndot('njfb,mnef->mbej', t2, self.get_ERI(L, 'oovv'), prefactor=0.5))

        Wmbej = careful_sum(terms, self.precision)

        return Wmbej

    # This intermediate appaears in the spin factorization of Wmbej terms.
    def build_Wmbje(self, ERI, t1, t2):
        terms = []
        terms.append(-1.0 * (self.get_ERI(ERI, 'ovov').copy()))
        terms.append(ndot('jf,mbfe->mbje', t1, self.get_ERI(ERI, 'ovvv'), prefactor=-1.0))
        terms.append(ndot('nb,mnje->mbje', t1, self.get_ERI(ERI, 'ooov')))
        tmp = (0.5 * t2)
        tmp += np.einsum('jf,nb->jnfb', t1, t1)
        terms.append(ndot('jnfb,mnfe->mbje', tmp, self.get_ERI(ERI, 'oovv')))

        Wmbje = careful_sum(terms, self.precision)

        return Wmbje

    # This intermediate is required to build second term of 0.5 * tau_ijef * Wabef,
    # as explicit construction of Wabef is avoided here.
    def build_Zmbij(self, ERI, t1, t2):
        Zmbij = ndot('mbef,ijef->mbij', self.get_ERI(ERI, 'ovvv'), self.build_tau(t1, t2))

        return Zmbij

    def r_T1(self, F, ERI, L, t1, t2, Fae, Fme, Fmi):
        terms = []
        terms.append(self.get_F(F, 'ov').copy())
        terms.append(ndot('ie,ae->ia', t1, Fae))
        terms.append(ndot('ma,mi->ia', t1, Fmi, prefactor=-1.0))
        t2_spinad = 2.0 * t2 - t2.swapaxes(2,3)
        terms.append(ndot('imae,me->ia', t2_spinad, Fme))
        terms.append(ndot('nf,nafi->ia', t1, self.get_ERI(L, 'ovvo')))
        terms.append(ndot('mief,maef->ia', t2_spinad, self.get_ERI(ERI, 'ovvv')))
        terms.append(ndot('mnae,nmei->ia', t2, self.get_ERI(L, 'oovo'), prefactor=-1.0))

        r_T1 = careful_sum(terms, self.precision)

        return r_T1

    def r_T2(self, F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij):

        terms = []
        if self.precision == "mixed":
            terms.append(0.5 * self.ERI64)
        else:
            terms.append(0.5 * self.get_ERI(ERI, 'oovv').copy())
        terms.append(ndot('ijae,be->ijab', t2, Fae))
        tmp = ndot('mb,me->be', t1, Fme)
        terms.append(ndot('ijae,be->ijab', t2, tmp, prefactor=-0.5))
        terms.append(ndot('imab,mj->ijab', t2, Fmi, prefactor=-1.0))
        tmp = ndot('je,me->jm', t1, Fme)
        terms.append(ndot('imab,jm->ijab', t2, tmp, prefactor=-0.5))
        tmp = self.build_tau(t1, t2)
        terms.append(ndot('mnab,mnij->ijab', tmp, Wmnij, prefactor=0.5))
        terms.append(ndot('ijef,abef->ijab', tmp, self.get_ERI(ERI, 'vvvv'), prefactor=0.5))
        terms.append(ndot('ma,mbij->ijab', t1, Zmbij, prefactor=-1.0))
        tmp = t2 - t2.swapaxes(2, 3)
        terms.append(ndot('imae,mbej->ijab', tmp, Wmbej))
        tmp = Wmbej + Wmbje.swapaxes(2, 3)
        terms.append(ndot('imae,mbej->ijab', t2, tmp))
        terms.append(ndot('mjae,mbie->ijab', t2, Wmbje))
        tmp = ndot('ie,ma->imea', t1, t1)
        terms.append(ndot('imea,mbej->ijab', tmp, self.get_ERI(ERI, 'ovvo'), prefactor=-1.0))
        tmp = ndot('ie,mb->imeb', t1, t1)
        terms.append(ndot('imeb,maje->ijab', tmp, self.get_ERI(ERI, 'ovov'), prefactor=-1.0))
        terms.append(ndot('ie,abej->ijab', t1, self.get_ERI(ERI, 'vvvo')))
        terms.append(ndot('ma,mbij->ijab', t1, self.get_ERI(ERI, 'ovoo'), prefactor=-1.0))

        r_T2 = careful_sum(terms, self.precision)
        r_T2 += r_T2.swapaxes(0, 1).swapaxes(2, 3)

        return r_T2

    # Assumes canonical RHF orbitals
    def compute_corr_energy(self, L, t1, t2):
        return np.einsum('ijab,ijab->', self.build_tau(t1, t2), self.get_ERI(L, 'oovv'))

    def compute_energy(self,
                       e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        ccsd_tstart = time.time()

        # Integrals and denominators
        F = self.F
        L = self.L
        ERI = self.ERI
        Dia = self.Dia
        Dijab = self.Dijab
        if self.precision == "mixed":
            F64 = self.F64
            L64 = self.L64

        # Compute MP2 energy
        # This is float64 for mixed- and double-precision; float32 for single
        if self.precision == "mixed":
            CCSDcorr_E_old = self.compute_corr_energy(L64, self.t1, self.t2)
        else:
            CCSDcorr_E_old = self.compute_corr_energy(L, self.t1, self.t2)

        print("CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  MP2" % (0, np.real(CCSDcorr_E_old), np.real(-CCSDcorr_E_old)))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.t1, self.t2, max_diis)

        # Iterations
        for CCSD_iter in range(1, maxiter + 1):
            if self.precision == "single" or self.precision == "mixed":
                t1 = np.float32(self.t1)
                t2 = np.float32(self.t2)
            else:
                t1 = self.t1
                t2 = self.t2

            # build intermediates
            Fae = self.build_Fae(F, L, t1, t2)
            Fme = self.build_Fme(F, L, t1)
            Fmi = self.build_Fmi(F, L, t1, t2)
            Wmnij = self.build_Wmnij(ERI, t1, t2)
            Wmbej = self.build_Wmbej(ERI, L, t1, t2)
            Wmbje = self.build_Wmbje(ERI, t1, t2)
            Zmbij = self.build_Zmbij(ERI, t1, t2)
            if self.precision == "single" or self.precision == "mixed":
                Fae = np.float32(Fae)
                Fme = np.float32(Fme)
                Fmi = np.float32(Fmi)
                Wmnij = np.float32(Wmnij)
                Wmbej = np.float32(Wmbej)
                Wmbje = np.float32(Wmbje)
                Zmbij = np.float32(Zmbij)

            # build residuals and update the amplitues
            r1 = self.r_T1(F, ERI, L, t1, t2, Fae, Fme, Fmi)
            r2 = self.r_T2(F, ERI, L, t1, t2, Fae, Fme, Fmi, Wmnij, Wmbej, Wmbje, Zmbij)
            self.t1 += r1 / Dia
            self.t2 += r2 / Dijab

            rms = np.einsum('ia,ia->', r1 / Dia, r1 / Dia)
            rms += np.einsum('ijab,ijab->', r2 / Dijab, r2 / Dijab)
            rms = np.sqrt(rms)

            # Compute CCSD correlation energy
            if self.precision == "mixed":
                CCSDcorr_E = self.compute_corr_energy(self.L64, self.t1, self.t2)
            else:
                CCSDcorr_E = self.compute_corr_energy(L, self.t1, self.t2)

            # Print CCSD iteration information
            print('CCSD Iter %3d: CCSD Ecorr = %.15f  dE = % .5E  rms = % .5E  DIIS = %d' % (
            CCSD_iter, CCSDcorr_E, CCSDcorr_E - CCSDcorr_E_old, rms, diis_object.diis_size))

            # Check convergence
            if (abs(CCSDcorr_E - CCSDcorr_E_old) < e_conv and rms < r_conv):
                print('\nCCSD has converged in %.3f seconds!' % (time.time() - ccsd_tstart))
                return CCSDcorr_E

            # Update old energy
            CCSDcorr_E_old = CCSDcorr_E

            #  Add the new error vector
            diis_object.add_error_vector(self.t1, self.t2)

            if CCSD_iter >= start_diis:
                self.t1, self.t2 = diis_object.extrapolate(self.t1, self.t2)

