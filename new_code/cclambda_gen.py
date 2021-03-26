# lambda-amplitudes l1, l2 in double-precision

import psi4
import time
import numpy as np
from utils import ndot
from utils import helper_diis

np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

def careful_sum(terms, precision):
    if precision == "single":
        prec='float32'
    elif precision == "double" or precision == "mixed":
        prec='float64'

    res = np.zeros_like(terms[0], dtype=prec)
    for arr in terms:
        if precision == "mixed": # cast each term to float64 before summing
            arr = arr.astype(prec)
        res = np.add(res, arr, dtype=prec)

    return res

class ccLambda(object):
    def __init__(self, ccsd, precision):
        #copy from ccsd object
        self.nocc  = ccsd.nocc
        self.nvirt = ccsd.nvirt
        self.nmo = ccsd.nmo
        self.ERI = ccsd.ERI
        self.L = ccsd.L
        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {
            'o': self.slice_o,
            'v': self.slice_v,
            'a': self.slice_a
        }
        self.Dia    = ccsd.Dia
        self.Dijab  = ccsd.Dijab
        self.t1     = ccsd.t1
        self.t2     = ccsd.t2
        self.F      = ccsd.F

        if precision != "double" and precision != "single" and precision != "mixed":
            raise Exception("Unknown option for precision:", precision)
        self.precision = precision
        print("Using %s precision." % precision)

        if self.precision=="mixed":
            self.F64 = self.F.copy()
            self.L64 = self.L.copy()
            self.ERI64 = self.ERI.copy()

        if self.precision == "single" or self.precision == "mixed":
            self.ERI = np.float32(self.ERI)
            self.L = np.float32(self.L)
            self.F = np.float32(self.F)

        # Initial guesses of l1 and l2
        self.l1 = 2 * self.t1.copy()
        self.l2 = 4 * self.t2.copy()
        self.l2 -= 2 * self.t2.swapaxes(2,3)

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

    def build_Goo(self, t2, l2):
        Goo = 0
        Goo += ndot('mjab,ijab->mi', t2, l2)
        return Goo

    def build_Gvv(self, t2, l2):
        Gvv = 0
        Gvv -= ndot('ijab,ijeb->ae', l2, t2)
        return Gvv

    def build_Loovv(self, L):
        Loovv = self.get_ERI(L, 'oovv').copy()
        #Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
        return Loovv

    def build_Looov(self, ERI):
        tmp = self.get_ERI(ERI, 'ooov').copy()
        Looov = 2.0 * tmp - tmp.swapaxes(0, 1)
        return Looov

    def build_Lvovv(self, L):
        Lvovv = self.get_ERI(L, 'vovv').copy()
        #Lvovv = 2.0 * tmp - tmp.swapaxes(2, 3)
        return Lvovv

    def build_tau(self, t1, t2):
        ttau = t2.copy()
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        ttau += tmp
        return ttau

    # F and W are the one and two body intermediates which appear in the CCSD
    # T1 and T2 equations. Please refer to ccenergy file for more details.

    def build_Hov(self, F, L, t1):
        """ <m|Hbar|e> = F_me = f_me + t_nf <mn||ef> """
        terms = []
        terms.append(self.get_F(F, 'ov').copy())
        Loovv = self.build_Loovv(L)
        terms.append(ndot('nf,mnef->me', t1, Loovv))

        Hov = careful_sum(terms, self.precision)

        return Hov

    def build_Hoo(self, F, ERI, L, t1, t2):
        """<m|Hbar|i> = F_mi + 0.5 * t_ie F_me = f_mi + t_ie f_me + t_ne <mn||ie> + tau_inef <mn||ef> """
        terms = []
        terms.append(self.get_F(F, 'oo').copy())
        terms.append(ndot('ie,me->mi', t1, self.get_F(F, 'ov')))
        terms.append(ndot('ne,mnie->mi', t1, self.build_Looov(ERI)))
        terms.append(ndot('inef,mnef->mi', self.build_tau(t1, t2), self.build_Loovv(L)))

        Hoo = careful_sum(terms, self.precision)

        return Hoo

    def build_Hvv(self, F, L, t1, t2):
        """<a|Hbar|e> = F_ae - 0.5 * t_ma F_me = f_ae - t_ma f_me + t_mf <am||ef> - tau_mnfa <mn||fe>"""
        terms = []
        terms.append(self.get_F(F, 'vv').copy())
        terms.append(ndot('ma,me->ae', t1, self.get_F(F, 'ov'), prefactor=-1.0))
        terms.append(ndot('mf,amef->ae', t1, self.build_Lvovv(L)))
        terms.append(ndot('mnfa,mnfe->ae', self.build_tau(t1, t2), self.build_Loovv(L), prefactor=-1.0))

        Hvv = careful_sum(terms, self.precision)

        return Hvv

    def build_Hoooo(self, ERI, t1, t2):
        """<mn|Hbar|ij> = W_mnij + 0.25 * tau_ijef <mn||ef> = <mn||ij> + P(ij) t_je <mn||ie> + 0.5 * tau_ijef <mn||ef>"""
        terms  =[]
        terms.append(self.get_ERI(ERI, 'oooo').copy())
        terms.append(ndot('je,mnie->mnij', t1, self.get_ERI(ERI, 'ooov')))
        terms.append(ndot('ie,mnej->mnij', t1, self.get_ERI(ERI, 'oovo')))
        terms.append(ndot('ijef,mnef->mnij', self.build_tau(t1, t2), self.get_ERI(ERI, 'oovv')))

        Hoooo = careful_sum(terms,self.precision)

        return Hoooo

    def build_Hvvvv(self, ERI, t1, t2):
        """<ab|Hbar|ef> = W_abef + 0.25 * tau_mnab <mn||ef> = <ab||ef> - P(ab) t_mb <am||ef> + 0.5 * tau_mnab <mn||ef>"""
        terms = []
        terms.append(self.get_ERI(ERI, 'vvvv').copy())
        terms.append(ndot('mb,amef->abef', t1, self.get_ERI(ERI, 'vovv'), prefactor=-1.0))
        terms.append(ndot('ma,bmfe->abef', t1, self.get_ERI(ERI, 'vovv'), prefactor=-1.0))
        terms.append(ndot('mnab,mnef->abef', self.build_tau(t1, t2), self.get_ERI(ERI, 'oovv')))

        Hvvvv = careful_sum(terms, self.precision)

        return Hvvvv

    def build_Hvovv(self, ERI, t1):
        """ <am|Hbar|ef> = <am||ef> - t_na <nm||ef> """
        terms = []
        terms.append(self.get_ERI(ERI, 'vovv').copy())
        terms.append(ndot('na,nmef->amef', t1, self.get_ERI(ERI, 'oovv'), prefactor=-1.0))

        Hvovv = careful_sum(terms, self.precision)

        return Hvovv

    def build_Hooov(self, ERI, t1):
        """ <mn|Hbar|ie> = <mn||ie> + t_if <mn||fe> """
        terms = []
        terms.append(self.get_ERI(ERI, 'ooov').copy())
        terms.append(ndot('if,mnfe->mnie', t1, self.get_ERI(ERI, 'oovv')))

        Hooov = careful_sum(terms, self.precision)

        return Hooov

    def build_Hovvo(self, ERI, L, t1, t2):
        """<mb|Hbar|ej> = W_mbej - 0.5 * t_jnfb <mn||ef> = <mb||ej> + t_jf <mb||ef>- t_nb <mn||ej> - (t_jnfb + t_jf t_nb) <nm||fe>"""
        terms = []
        terms.append(self.get_ERI(ERI, 'ovvo').copy())
        terms.append(ndot('jf,mbef->mbej', t1, self.get_ERI(ERI, 'ovvv')))
        terms.append(ndot('nb,mnej->mbej', t1, self.get_ERI(ERI, 'oovo'), prefactor=-1.0))
        terms.append(ndot('jnfb,nmfe->mbej', self.build_tau(t1, t2), self.get_ERI(ERI, 'oovv'), prefactor=-1.0))
        terms.append(ndot('jnbf,nmfe->mbej', t2, self.build_Loovv(L)))

        Hovvo = careful_sum(terms,self.precision)

        return Hovvo

    def build_Hovov(self, ERI, t1, t2):
        """<mb|Hbar|je> = - <mb|Hbar|ej> = <mb||je> + t_jf <bm||ef> - t_nb <mn||je> - (t_jnfb + t_jf t_nb) <nm||ef>"""
        terms = []
        terms.append(self.get_ERI(ERI, 'ovov').copy())
        terms.append(ndot('jf,bmef->mbje', t1, self.get_ERI(ERI, 'vovv')))
        terms.append(ndot('nb,mnje->mbje', t1, self.get_ERI(ERI, 'ooov'), prefactor=-1.0))
        terms.append(ndot('jnfb,nmef->mbje', self.build_tau(t1, t2), self.get_ERI(ERI, 'oovv'), prefactor=-1.0))

        Hovov = careful_sum(terms, self.precision)

        return Hovov

    def build_Hvvvo(self, F, ERI, L, t1, t2):
        """<ab|Hbar|ei> = <ab||ei> - F_me t_miab + t_if Wabef + 0.5 * tau_mnab <mn||ei>- P(ab) t_miaf <mb||ef> - P(ab) t_ma {<mb||ei> - t_nibf <mn||ef>}"""
        terms = []
        # <ab||ei>
        terms.append(self.get_ERI(ERI, 'vvvo').copy())

        # - Fme t_miab

        terms.append(ndot('me,miab->abei', self.get_F(F, 'ov'), t2, prefactor=-1.0))
        tmp = ndot('mnfe,mf->ne', self.build_Loovv(L), t1)
        terms.append(ndot('niab,ne->abei', t2, tmp, prefactor=-1.0))

        # t_if Wabef

        terms.append(ndot('if,abef->abei', t1, self.get_ERI(ERI, 'vvvv')))
        tmp = ndot('if,ma->imfa', t1, t1)
        terms.append(ndot('imfa,mbef->abei', tmp, self.get_ERI(ERI, 'ovvv'), prefactor=-1.0))
        terms.append(ndot('imfb,amef->abei', tmp, self.get_ERI(ERI, 'vovv'), prefactor=-1.0))
        tmp = ndot('mnef,if->mnei', self.get_ERI(ERI, 'oovv'), t1)
        terms.append(ndot('mnab,mnei->abei', t2, tmp))
        tmp = ndot('if,ma->imfa', t1, t1)
        tmp1 = ndot('mnef,nb->mbef', self.get_ERI(ERI, 'oovv'), t1)
        terms.append(ndot('imfa,mbef->abei', tmp, tmp1))

        # 0.5 * tau_mnab <mn||ei>

        terms.append(ndot('mnab,mnei->abei', self.build_tau(t1, t2), self.get_ERI(ERI, 'oovo')))

        # - P(ab) t_miaf <mb||ef>

        terms.append(ndot('imfa,mbef->abei', t2, self.get_ERI(ERI, 'ovvv'), prefactor=-1.0))
        terms.append(ndot('imfb,amef->abei', t2, self.get_ERI(ERI, 'vovv'), prefactor=-1.0))
        terms.append(ndot('mifb,amef->abei', t2, self.build_Lvovv(L)))

        # - P(ab) t_ma <mb||ei>

        terms.append(ndot('mb,amei->abei', t1, self.get_ERI(ERI, 'vovo'), prefactor=-1.0))
        terms.append(ndot('ma,bmie->abei', t1, self.get_ERI(ERI, 'voov'), prefactor=-1.0))

        # P(ab) t_ma * t_nibf <mn||ef>

        tmp = ndot('mnef,ma->anef', self.get_ERI(ERI, 'oovv'), t1)
        terms.append(ndot('infb,anef->abei', t2, tmp))
        tmp = ndot('mnef,ma->nafe', self.build_Loovv(L), t1)
        terms.append(ndot('nifb,nafe->abei', t2, tmp, prefactor=-1.0))
        tmp = ndot('nmef,mb->nefb', self.get_ERI(ERI, 'oovv'), t1)
        terms.append(ndot('niaf,nefb->abei', t2, tmp))

        Hvvvo = careful_sum(terms, self.precision)

        return Hvvvo

    def build_Hovoo(self, F, ERI, L, t1, t2):
        """<mb|Hbar|ij> = <mb||ij> - Fme t_ijbe - t_nb Wmnij + 0.5 * tau_ijef <mb||ef> + P(ij) t_jnbe <mn||ie> + P(ij) t_ie {<mb||ej> - t_njbf <mn||ef>}"""
        terms = []
        # <mb||ij>

        terms.append(self.get_ERI(ERI, 'ovoo').copy())

        # - Fme t_ijbe

        terms.append(ndot('me,ijeb->mbij', self.get_F(F, 'ov'), t2))
        tmp = ndot('mnef,nf->me', self.build_Loovv(L), t1)
        terms.append(ndot('me,ijeb->mbij', tmp, t2))

        # - t_nb Wmnij

        terms.append(ndot('nb,mnij->mbij', t1, self.get_ERI(ERI, 'oooo'), prefactor=-1.0))
        tmp = ndot('ie,nb->ineb', t1, t1)
        terms.append(ndot('ineb,mnej->mbij', tmp, self.get_ERI(ERI, 'oovo'), prefactor=-1.0))
        terms.append(ndot('jneb,mnie->mbij', tmp, self.get_ERI(ERI, 'ooov'), prefactor=-1.0))
        tmp = ndot('nb,mnef->mefb', t1, self.get_ERI(ERI, 'oovv'))
        terms.append(ndot('ijef,mefb->mbij', t2, tmp, prefactor=-1.0))
        tmp = ndot('ie,jf->ijef', t1, t1)
        tmp1 = ndot('nb,mnef->mbef', t1, self.get_ERI(ERI, 'oovv'))
        terms.append(ndot('mbef,ijef->mbij', tmp1, tmp, prefactor=-1.0))

        # 0.5 * tau_ijef <mb||ef>

        terms.append(ndot('ijef,mbef->mbij', self.build_tau(t1, t2),
                      self.get_ERI(ERI, 'ovvv')))

        # P(ij) t_jnbe <mn||ie>

        terms.append(ndot('ineb,mnej->mbij', t2, self.get_ERI(ERI, 'oovo'), prefactor=-1.0))
        terms.append(ndot('jneb,mnie->mbij', t2, self.get_ERI(ERI, 'ooov'), prefactor=-1.0))
        terms.append(ndot('jnbe,mnie->mbij', t2, self.build_Looov(ERI)))

        # P(ij) t_ie <mb||ej>

        terms.append(ndot('je,mbie->mbij', t1, self.get_ERI(ERI, 'ovov')))
        terms.append(ndot('ie,mbej->mbij', t1, self.get_ERI(ERI, 'ovvo')))

        # - P(ij) t_ie * t_njbf <mn||ef>

        tmp = ndot('ie,mnef->mnif', t1, self.get_ERI(ERI, 'oovv'))
        terms.append(ndot('jnfb,mnif->mbij', t2, tmp, prefactor=-1.0))
        tmp = ndot('mnef,njfb->mejb', self.build_Loovv(L), t2)
        terms.append(ndot('mejb,ie->mbij', tmp, t1))
        tmp = ndot('je,mnfe->mnfj', t1, self.get_ERI(ERI, 'oovv'))
        terms.append(ndot('infb,mnfj->mbij', t2, tmp, prefactor=-1.0))

        Hovoo = careful_sum(terms, self.precision)

        return Hovoo

    def r_l1(self, F, ERI, L, t1, t2, l1, l2):
        if self.precision == "mixed":
            Hvv = np.float32(self.build_Hvv(F, L, t1, t2))
            Hoo = np.float32(self.build_Hoo(F, ERI, L, t1, t2))
            Hovvo = np.float32(self.build_Hovvo(ERI, L, t1, t2))
            Hovov = np.float32(self.build_Hovov(ERI, t1, t2))
            Hvvvo = np.float32(self.build_Hvvvo(F, ERI, L, t1, t2))
            Hovoo = np.float32(self.build_Hovoo(F, ERI, L, t1, t2))
            Hvovv = np.float32(self.build_Hvovv(ERI, t1))
            Hooov = np.float32(self.build_Hooov(ERI, t1))

        else:
            Hvv = self.build_Hvv(F, L, t1, t2)
            Hoo = self.build_Hoo(F, ERI, L, t1, t2)
            Hovvo = self.build_Hovvo(ERI, L, t1, t2)
            Hovov = self.build_Hovov(ERI, t1, t2)
            Hvvvo = self.build_Hvvvo(F, ERI, L, t1, t2)
            Hovoo = self.build_Hovoo(F, ERI, L, t1, t2)
            Hvovv = self.build_Hvovv(ERI, t1)
            Hooov = self.build_Hooov(ERI, t1)
        terms = []
        terms.append(2.0 * self.build_Hov(F, L, t1).copy())
        terms.append(ndot('ie,ea->ia', l1, Hvv))
        terms.append(ndot('im,ma->ia', Hoo, l1, prefactor=-1.0))
        Hovvo_spinad = 2.0 * Hovvo - Hovov.swapaxes(2,3)
        terms.append(ndot('ieam,me->ia', Hovvo_spinad, l1))
        terms.append(ndot('imef,efam->ia', l2, Hvvvo))
        terms.append(ndot('iemn,mnae->ia', Hovoo, l2, prefactor=-1.0))
        Hvovv_spinad = 2.0 * Hvovv - Hvovv.swapaxes(2,3)
        terms.append(ndot('eifa,ef->ia', Hvovv_spinad, self.build_Gvv(t2, l2), prefactor=-1.0))
        Hooov_spinad = 2.0 * Hooov - Hooov.swapaxes(0,1)
        terms.append(ndot('mina,mn->ia', Hooov_spinad, self.build_Goo(t2, l2), prefactor=-1.0))

        r_l1 = careful_sum(terms, self.precision)

        return r_l1

    def r_l2(self, F, ERI, L, t1, t2, l1, l2):
        if self.precision == "mixed":
            Hov = np.float32(self.build_Hov(F, L, t1))
            Hvv = np.float32(self.build_Hvv(F, L, t1, t2))
            Hoo = np.float32(self.build_Hoo(F, ERI, L, t1, t2))
            Hoooo = np.float32(self.build_Hoooo(ERI, t1, t2))
            Hvvvv = np.float32(self.build_Hvvvv(ERI, t1, t2))
            Hovvo = np.float32(self.build_Hovvo(ERI, L, t1, t2))
            Hovov = np.float32(self.build_Hovov(ERI, t1, t2))
            Hvovv = np.float32(self.build_Hvovv(ERI, t1))
            Hooov = np.float32(self.build_Hooov(ERI, t1))

        else:
            Hov = self.build_Hov(F, L, t1)
            Hvv = self.build_Hvv(F, L, t1, t2)
            Hoo = self.build_Hoo(F, ERI, L, t1, t2)
            Hoooo = self.build_Hoooo(ERI, t1, t2)
            Hvvvv = self.build_Hvvvv(ERI, t1, t2)
            Hovvo = self.build_Hovvo(ERI, L, t1, t2)
            Hovov = self.build_Hovov(ERI, t1, t2)
            Hvovv = self.build_Hvovv(ERI, t1)
            Hooov = self.build_Hooov(ERI, t1)
        terms = []
        if self.precision == "mixed":
            terms.append(self.build_Loovv(self.L64).copy())
        else:
            terms.append(self.build_Loovv(L).copy())
        tmp = ndot('ia,jb->ijab', l1, Hov, prefactor=2.0)
        tmp -= ndot('ja,ib->ijab', l1, Hov)
        terms.append(tmp)
        terms.append(ndot('ijeb,ea->ijab', l2, Hvv))
        terms.append(ndot('im,mjab->ijab', Hoo, l2, prefactor=-1.0))
        terms.append(ndot('ijmn,mnab->ijab', Hoooo, l2, prefactor=0.5))
        terms.append(ndot('ijef,efab->ijab', l2, Hvvvv, prefactor=0.5))
        Hvovv_spinad = 2.0 * Hvovv - Hvovv.swapaxes(2,3)
        terms.append(ndot('ie,ejab->ijab', l1, Hvovv_spinad))
        Hooov_spinad = 2.0 * Hooov - Hooov.swapaxes(0,1)
        terms.append(ndot('mb,jima->ijab', l1, Hooov_spinad, prefactor=-1.0))
        Hovvo_spinad = 2.0 * Hovvo - Hovov.swapaxes(2,3)
        terms.append(ndot('ieam,mjeb->ijab', Hovvo_spinad, l2))
        terms.append(ndot('mibe,jema->ijab', l2, Hovov, prefactor=-1.0))
        terms.append(ndot('mieb,jeam->ijab', l2, Hovvo, prefactor=-1.0))
        terms.append(ndot('ijeb,ae->ijab', self.build_Loovv(L), self.build_Gvv(t2, l2)))
        terms.append(ndot('mi,mjab->ijab', self.build_Goo(t2, l2), self.build_Loovv(L), prefactor=-1.0))

        r_l2 = careful_sum(terms, self.precision)

        return r_l2

    def update_l(self):
        old_l2 = self.l2.copy()
        old_l1 = self.l1.copy()

        rl1 = self.r_l1(self.F, self.ERI, self.L, self.t1, self.t2, self.l1, self.l2)
        rl2 = self.r_l2(self.F, self.ERI, self.L, self.t1, self.t2, self.l1, self.l2)
        # update l1 and l2 amplitudes
        self.l1 += np.real(rl1) / np.real(self.Dia)
        # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
        tmp = np.real(rl2) / np.real(self.Dijab)
        self.l2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)

        # calculate rms from the residual
        rms = 2.0 * np.einsum('ia,ia->', old_l1 - self.l1, old_l1 - self.l1)
        rms += np.einsum('ijab,ijab->', old_l2 - self.l2, old_l2 - self.l2)
        return np.sqrt(rms)

    def pseudoenergy(self, ERI, l2):
        pseudoenergy = 0
        pseudoenergy += ndot('ijab,ijab->', self.get_ERI(ERI, 'oovv'), l2, prefactor=0.5)
        return pseudoenergy

    def compute_lambda(self,
                       e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        cclambda_tstart = time.time()
        if self.precision == "mixed":
            pseudoenergy_old = self.pseudoenergy(self.ERI64, self.l2)
        else:
            pseudoenergy_old = self.pseudoenergy(self.ERI, self.l2)
        print(
            "CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E" % (0, np.real(pseudoenergy_old), np.real(-pseudoenergy_old)))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.l1, self.l2, max_diis)

        # Iterate
        for CCLAMBDA_iter in range(1, maxiter + 1):
            rms_l = self.update_l()
            # Compute pseudoenergy

            if self.precision == "mixed":
                pseudo_energy = self.pseudoenergy(self.ERI64, self.l2)
            else:
                pseudo_energy = self.pseudoenergy(self.ERI, self.l2)

            # Print CCLAMBDA iteration information
            print('CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E   DIIS = %d'
                  % (CCLAMBDA_iter, np.real(pseudo_energy),
                     np.real(pseudo_energy - pseudoenergy_old), diis_object.diis_size))

            # Check convergence
            if (rms_l < r_conv):
                print('\nCCLAMBDA has converged in %.3f seconds!' % (time.time() - cclambda_tstart))
                return pseudo_energy

            # Update old pseudoenergy
            pseudoenergy_old = pseudo_energy

            #  Add the new error vector
            diis_object.add_error_vector(self.l1, self.l2)
            if CCLAMBDA_iter >= start_diis:
                self.l1, self.l2 = diis_object.extrapolate(self.l1, self.l2)






