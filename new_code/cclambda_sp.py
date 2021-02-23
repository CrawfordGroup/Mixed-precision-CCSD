# lambda-amplitudes l1, l2 in mixed-precision

import psi4
import time
import numpy as np
from utils import ndot
from opt_einsum import contract
from utils import helper_diis

np.set_printoptions(precision=5, linewidth=200, threshold=200, suppress=True)

class ccLambda_sp(object):
    def __init__(self, ccsd):
        #copy from ccsd object
        self.nocc  = ccsd.nocc
        self.nvirt = ccsd.nvirt
        self.nmo = ccsd.nmo
        self.MO = ccsd.MO
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

        # Initial guesses of l1 and l2
        self.l1 = 2 * self.t1.copy()
        self.l2 = 4 * self.t2.copy()
        self.l2 -= 2 * self.t2.swapaxes(2,3)

        # Single-precision
        self.MOsp = ccsd.MOsp
        self.Fsp = ccsd.Fsp
        self.t1_sp = ccsd.t1_sp
        self.t2_sp = ccsd.t2_sp
        self.l1_sp = np.float32(self.l1)
        self.l2_sp = np.float32(self.l2)

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

    # F is time-dependent during the propagation*
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

    def build_Loovv(self):
        tmp = self.get_MO_sp('oovv').copy()
        Loovv = 2.0 * tmp - tmp.swapaxes(2, 3)
        return Loovv

    def build_Looov(self):
        tmp = self.get_MO_sp('ooov').copy()
        Looov = 2.0 * tmp - tmp.swapaxes(0, 1)
        return Looov

    def build_Lvovv(self):
        tmp = self.get_MO_sp('vovv').copy()
        Lvovv = 2.0 * tmp - tmp.swapaxes(2, 3)
        return Lvovv

    def build_tau(self, t1, t2):
        ttau = t2.copy()
        tmp = np.einsum('ia,jb->ijab', t1, t1)
        ttau += tmp
        return ttau

    # F and W are the one and two body intermediates which appear in the CCSD
    # T1 and T2 equations. Please refer to ccenergy file for more details.

    def build_Hov(self, F, t1):
        """ <m|Hbar|e> = F_me = f_me + t_nf <mn||ef> """
        Hov = self.get_F(F, 'ov').copy()
        Loovv = self.build_Loovv()
        Hov += ndot('nf,mnef->me', t1, Loovv)
        return Hov

    def build_Hoo(self, F, t1, t2):
        """<m|Hbar|i> = F_mi + 0.5 * t_ie F_me = f_mi + t_ie f_me + t_ne <mn||ie> + tau_inef <mn||ef> """
        Hoo = self.get_F(F, 'oo').copy()
        Hoo += ndot('ie,me->mi', t1, self.get_F(F, 'ov'))
        Looov = self.build_Looov()
        Hoo += ndot('ne,mnie->mi', t1, Looov)
        Loovv = self.build_Loovv()
        Hoo += ndot('inef,mnef->mi', self.build_tau(t1, t2), Loovv)
        return Hoo

    def build_Hvv(self, F, t1, t2):
        """<a|Hbar|e> = F_ae - 0.5 * t_ma F_me = f_ae - t_ma f_me + t_mf <am||ef> - tau_mnfa <mn||fe>"""
        Hvv = self.get_F(F, 'vv').copy()
        Hvv -= ndot('ma,me->ae', t1, self.get_F(F, 'ov'))
        Lvovv = self.build_Lvovv()
        Hvv += ndot('mf,amef->ae', t1, Lvovv)
        Loovv = self.build_Loovv()
        Hvv -= ndot('mnfa,mnfe->ae', self.build_tau(t1, t2), Loovv)
        return Hvv

    def build_Hoooo(self, t1, t2):
        """<mn|Hbar|ij> = W_mnij + 0.25 * tau_ijef <mn||ef> = <mn||ij> + P(ij) t_je <mn||ie> + 0.5 * tau_ijef <mn||ef>"""
        Hoooo = self.get_MO_sp('oooo').copy()
        Hoooo += ndot('je,mnie->mnij', t1, self.get_MO_sp('ooov'))
        Hoooo += ndot('ie,mnej->mnij', t1, self.get_MO_sp('oovo'))
        Hoooo += ndot('ijef,mnef->mnij', self.build_tau(t1, t2), self.get_MO_sp('oovv'))
        return Hoooo

    def build_Hvvvv(self, t1, t2):
        """<ab|Hbar|ef> = W_abef + 0.25 * tau_mnab <mn||ef> = <ab||ef> - P(ab) t_mb <am||ef> + 0.5 * tau_mnab <mn||ef>"""
        Hvvvv = self.get_MO_sp('vvvv').copy()
        Hvvvv -= ndot('mb,amef->abef', t1, self.get_MO_sp('vovv'))
        Hvvvv -= ndot('ma,bmfe->abef', t1, self.get_MO_sp('vovv'))
        Hvvvv += ndot('mnab,mnef->abef', self.build_tau(t1, t2), self.get_MO_sp('oovv'))
        return Hvvvv

    def build_Hvovv(self, t1):
        """ <am|Hbar|ef> = <am||ef> - t_na <nm||ef> """
        Hvovv = self.get_MO_sp('vovv').copy()
        Hvovv -= ndot('na,nmef->amef', t1, self.get_MO_sp('oovv'))
        return Hvovv

    def build_Hooov(self, t1):
        """ <mn|Hbar|ie> = <mn||ie> + t_if <mn||fe> """
        Hooov = self.get_MO_sp('ooov').copy()
        Hooov += ndot('if,mnfe->mnie', t1, self.get_MO_sp('oovv'))
        return Hooov

    def build_Hovvo(self, t1, t2):
        """<mb|Hbar|ej> = W_mbej - 0.5 * t_jnfb <mn||ef> = <mb||ej> + t_jf <mb||ef>- t_nb <mn||ej> - (t_jnfb + t_jf t_nb) <nm||fe>"""
        Hovvo = self.get_MO_sp('ovvo').copy()
        Hovvo += ndot('jf,mbef->mbej', t1, self.get_MO_sp('ovvv'))
        Hovvo -= ndot('nb,mnej->mbej', t1, self.get_MO_sp('oovo'))
        Hovvo -= ndot('jnfb,nmfe->mbej', self.build_tau(t1, t2), self.get_MO_sp('oovv'))
        Loovv = self.build_Loovv()
        Hovvo += ndot('jnbf,nmfe->mbej', t2, Loovv)
        return Hovvo

    def build_Hovov(self, t1, t2):
        """<mb|Hbar|je> = - <mb|Hbar|ej> = <mb||je> + t_jf <bm||ef> - t_nb <mn||je> - (t_jnfb + t_jf t_nb) <nm||ef>"""
        Hovov = self.get_MO_sp('ovov').copy()
        Hovov += ndot('jf,bmef->mbje', t1, self.get_MO_sp('vovv'))
        Hovov -= ndot('nb,mnje->mbje', t1, self.get_MO_sp('ooov'))
        Hovov -= ndot('jnfb,nmef->mbje', self.build_tau(t1, t2), self.get_MO_sp('oovv'))
        return Hovov

    def build_Hvvvo(self, F, t1, t2):
        """<ab|Hbar|ei> = <ab||ei> - F_me t_miab + t_if Wabef + 0.5 * tau_mnab <mn||ei>- P(ab) t_miaf <mb||ef> - P(ab) t_ma {<mb||ei> - t_nibf <mn||ef>}"""
        # <ab||ei>

        Hvvvo = self.get_MO_sp('vvvo').copy()

        # - Fme t_miab

        Hvvvo -= ndot('me,miab->abei', self.get_F(F, 'ov'), t2)
        Loovv = self.build_Loovv()
        tmp = ndot('mnfe,mf->ne', Loovv, t1)
        Hvvvo -= ndot('niab,ne->abei', t2, tmp)

        # t_if Wabef

        Hvvvo += ndot('if,abef->abei', t1, self.get_MO_sp('vvvv'))
        tmp = ndot('if,ma->imfa', t1, t1)
        Hvvvo -= ndot('imfa,mbef->abei', tmp, self.get_MO_sp('ovvv'))
        Hvvvo -= ndot('imfb,amef->abei', tmp, self.get_MO_sp('vovv'))
        tmp = ndot('mnef,if->mnei', self.get_MO_sp('oovv'), t1)
        Hvvvo += ndot('mnab,mnei->abei', t2, tmp)
        tmp = ndot('if,ma->imfa', t1, t1)
        tmp1 = ndot('mnef,nb->mbef', self.get_MO_sp('oovv'), t1)
        Hvvvo += ndot('imfa,mbef->abei', tmp, tmp1)

        # 0.5 * tau_mnab <mn||ei>

        Hvvvo += ndot('mnab,mnei->abei', self.build_tau(t1, t2), self.get_MO_sp('oovo'))

        # - P(ab) t_miaf <mb||ef>

        Hvvvo -= ndot('imfa,mbef->abei', t2, self.get_MO_sp('ovvv'))
        Hvvvo -= ndot('imfb,amef->abei', t2, self.get_MO_sp('vovv'))
        Lvovv = self.build_Lvovv()
        Hvvvo += ndot('mifb,amef->abei', t2, Lvovv)

        # - P(ab) t_ma <mb||ei>

        Hvvvo -= ndot('mb,amei->abei', t1, self.get_MO_sp('vovo'))
        Hvvvo -= ndot('ma,bmie->abei', t1, self.get_MO_sp('voov'))

        # P(ab) t_ma * t_nibf <mn||ef>

        tmp = ndot('mnef,ma->anef', self.get_MO_sp('oovv'), t1)
        Hvvvo += ndot('infb,anef->abei', t2, tmp)
        Loovv = self.build_Loovv()
        tmp = ndot('mnef,ma->nafe', Loovv, t1)
        Hvvvo -= ndot('nifb,nafe->abei', t2, tmp)
        tmp = ndot('nmef,mb->nefb', self.get_MO_sp('oovv'), t1)
        Hvvvo += ndot('niaf,nefb->abei', t2, tmp)
        return Hvvvo

    def build_Hovoo(self, F, t1, t2):
        """<mb|Hbar|ij> = <mb||ij> - Fme t_ijbe - t_nb Wmnij + 0.5 * tau_ijef <mb||ef> + P(ij) t_jnbe <mn||ie> + P(ij) t_ie {<mb||ej> - t_njbf <mn||ef>}"""
        # <mb||ij>

        Hovoo = self.get_MO_sp('ovoo').copy()

        # - Fme t_ijbe

        Hovoo += ndot('me,ijeb->mbij', self.get_F(F, 'ov'), t2)
        Loovv = self.build_Loovv()
        tmp = ndot('mnef,nf->me', Loovv, t1)
        Hovoo += ndot('me,ijeb->mbij', tmp, t2)

        # - t_nb Wmnij

        Hovoo -= ndot('nb,mnij->mbij', t1, self.get_MO_sp('oooo'))
        tmp = ndot('ie,nb->ineb', t1, t1)
        Hovoo -= ndot('ineb,mnej->mbij', tmp, self.get_MO_sp('oovo'))
        Hovoo -= ndot('jneb,mnie->mbij', tmp, self.get_MO_sp('ooov'))
        tmp = ndot('nb,mnef->mefb', t1, self.get_MO_sp('oovv'))
        Hovoo -= ndot('ijef,mefb->mbij', t2, tmp)
        tmp = ndot('ie,jf->ijef', t1, t1)
        tmp1 = ndot('nb,mnef->mbef', t1, self.get_MO_sp('oovv'))
        Hovoo -= ndot('mbef,ijef->mbij', tmp1, tmp)

        # 0.5 * tau_ijef <mb||ef>

        Hovoo += ndot('ijef,mbef->mbij', self.build_tau(t1, t2),
                      self.get_MO_sp('ovvv'))

        # P(ij) t_jnbe <mn||ie>

        Hovoo -= ndot('ineb,mnej->mbij', t2, self.get_MO_sp('oovo'))
        Hovoo -= ndot('jneb,mnie->mbij', t2, self.get_MO_sp('ooov'))
        Looov = self.build_Looov()
        Hovoo += ndot('jnbe,mnie->mbij', t2, Looov)

        # P(ij) t_ie <mb||ej>

        Hovoo += ndot('je,mbie->mbij', t1, self.get_MO_sp('ovov'))
        Hovoo += ndot('ie,mbej->mbij', t1, self.get_MO_sp('ovvo'))

        # - P(ij) t_ie * t_njbf <mn||ef>

        tmp = ndot('ie,mnef->mnif', t1, self.get_MO_sp('oovv'))
        Hovoo -= ndot('jnfb,mnif->mbij', t2, tmp)
        tmp = ndot('mnef,njfb->mejb', Loovv, t2)
        Hovoo += ndot('mejb,ie->mbij', tmp, t1)
        tmp = ndot('je,mnfe->mnfj', t1, self.get_MO_sp('oovv'))
        Hovoo -= ndot('infb,mnfj->mbij', t2, tmp)
        return Hovoo

    # Use single-precision F, t1, t2, l1 and l2 (for the contractions)
    def r_l1(self, F, t1, t2, l1, l2):
        r_l1 = 2.0 * self.build_Hov(F, t1).copy()
        r_l1 += ndot('ie,ea->ia', l1, self.build_Hvv(F, t1, t2))
        r_l1 -= ndot('im,ma->ia', self.build_Hoo(F, t1, t2), l1)
        r_l1 += ndot('ieam,me->ia', self.build_Hovvo(t1, t2), l1, prefactor=2.0)
        r_l1 += ndot('iema,me->ia', self.build_Hovov(t1, t2), l1, prefactor=-1.0)
        r_l1 += ndot('imef,efam->ia', l2, self.build_Hvvvo(F, t1, t2))
        r_l1 -= ndot('iemn,mnae->ia', self.build_Hovoo(F, t1, t2), l2)
        r_l1 -= ndot('eifa,ef->ia', self.build_Hvovv(t1), self.build_Gvv(t2, l2), prefactor=2.0)
        r_l1 -= ndot('eiaf,ef->ia', self.build_Hvovv(t1), self.build_Gvv(t2, l2), prefactor=-1.0)
        r_l1 -= ndot('mina,mn->ia', self.build_Hooov(t1), self.build_Goo(t2, l2), prefactor=2.0)
        r_l1 -= ndot('imna,mn->ia', self.build_Hooov(t1), self.build_Goo(t2, l2), prefactor=-1.0)

        return r_l1

    def r_l2(self, F, t1, t2, l1, l2):
        r_l2 = self.build_Loovv().copy()
        r_l2 += ndot('ia,jb->ijab', l1, self.build_Hov(F, t1), prefactor=2.0)
        r_l2 -= ndot('ja,ib->ijab', l1, self.build_Hov(F, t1))
        r_l2 += ndot('ijeb,ea->ijab', l2, self.build_Hvv(F, t1, t2))
        r_l2 -= ndot('im,mjab->ijab', self.build_Hoo(F, t1, t2), l2)
        r_l2 += ndot('ijmn,mnab->ijab', self.build_Hoooo(t1, t2), l2, prefactor=0.5)
        r_l2 += ndot('ijef,efab->ijab', l2, self.build_Hvvvv(t1, t2), prefactor=0.5)
        r_l2 += ndot('ie,ejab->ijab', l1, self.build_Hvovv(t1), prefactor=2.0)
        r_l2 += ndot('ie,ejba->ijab', l1, self.build_Hvovv(t1), prefactor=-1.0)
        r_l2 -= ndot('mb,jima->ijab', l1, self.build_Hooov(t1), prefactor=2.0)
        r_l2 -= ndot('mb,ijma->ijab', l1, self.build_Hooov(t1), prefactor=-1.0)
        r_l2 += ndot('ieam,mjeb->ijab', self.build_Hovvo(t1, t2), l2, prefactor=2.0)
        r_l2 += ndot('iema,mjeb->ijab', self.build_Hovov(t1, t2), l2, prefactor=-1.0)
        r_l2 -= ndot('mibe,jema->ijab', l2, self.build_Hovov(t1, t2))
        r_l2 -= ndot('mieb,jeam->ijab', l2, self.build_Hovvo(t1, t2))
        r_l2 += ndot('ijeb,ae->ijab', self.build_Loovv(), self.build_Gvv(t2, l2))
        r_l2 -= ndot('mi,mjab->ijab', self.build_Goo(t2, l2), self.build_Loovv())

        return r_l2

    def update_l(self):
        old_l2 = self.l2.copy()
        old_l1 = self.l1.copy()

        rl1 = self.r_l1(self.Fsp, self.t1_sp, self.t2_sp, self.l1_sp, self.l2_sp)
        rl2 = self.r_l2(self.Fsp, self.t1_sp, self.t2_sp, self.l1_sp, self.l2_sp)
        # update l1 and l2 amplitudes
        self.l1 += np.float64(rl1) / self.Dia
        # Final r_l2_ijab = r_l2_ijab + r_l2_jiba
        tmp = np.float64(rl2) / self.Dijab
        self.l2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)
        # update single-precision l1, l2
        self.l1_sp = np.float32(self.l1)
        self.l2_sp = np.float32(self.l2)
        # calculate rms from the residual
        rms = 2.0 * np.einsum('ia,ia->', old_l1 - self.l1, old_l1 - self.l1)
        rms += np.einsum('ijab,ijab->', old_l2 - self.l2, old_l2 - self.l2)
        return np.sqrt(rms)

    def pseudoenergy(self, l2):
        pseudoenergy = 0
        pseudoenergy += ndot('ijab,ijab->', self.get_MO('oovv'), l2, prefactor=0.5)
        return pseudoenergy

    def compute_lambda(self,
                       e_conv=1e-7,
                       r_conv=1e-7,
                       maxiter=100,
                       max_diis=8,
                       start_diis=1):

        ### Start Iterations
        cclambda_tstart = time.time()
        pseudoenergy_old = self.pseudoenergy(self.l2)
        print(
            "CCLAMBDA Iteration %3d: pseudoenergy = %.15f   dE = % .5E" % (0, np.real(pseudoenergy_old), np.real(-pseudoenergy_old)))

        # Set up DIIS before iterations begin
        diis_object = helper_diis(self.l1, self.l2, max_diis)

        # Iterate
        for CCLAMBDA_iter in range(1, maxiter + 1):
            rms_l = self.update_l()
            # Compute pseudoenergy
            pseudo_energy = self.pseudoenergy(self.l2)

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






