# -*- coding: utf-8 -*-
"""
A simple python script to compute RHF-CCSD linear response function
for calculating properties like dipole polarizabilities, optical
rotations etc.
References:
- Equations and algoriths from [Koch:1991:3333], [Gwaltney:1996:189],
[Helgaker:2000], and [Crawford:xxxx]
1. A Whirlwind Introduction to Coupled Cluster Response Theory, T.D. Crawford, Private Notes,
   (pdf in the current directory).
2. H. Koch and P. Jørgensen, J. Chem. Phys. Volume 93, pp. 3333-3344 (1991).
3. S. R. Gwaltney, M. Nooijen and R.J. Bartlett, Chemical Physics Letters, 248, pp. 189-198 (1996).
4. Chapter 13, "Molecular Electronic-Structure Theory", Trygve Helgaker,
   Poul Jørgensen and Jeppe Olsen, John Wiley & Sons Ltd.
"""

__authors__ = "Zhe Wang"
__credits__ = ["Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"

import time
import numpy as np
import psi4
#import sys
#sys.path.append("../../../Coupled-Cluster/RHF")
from utils import ndot
from utils import helper_diis
from opt_einsum import contract

class CCPert(object):
    def __init__(self, name, pert, ccsd, cclambda, omega):

        # start of the ccpert class
        time_init = time.time()

        # Grabbing all the info from the wavefunctions passed
        self.pert = pert
        self.name = name
        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc
        self.mints = ccsd.mints
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        # Single-precision
        self.MOsp = ccsd.MOsp
        self.Fsp = ccsd.Fsp
        self.pert_sp = np.float32(self.pert)
        self.t1_sp = ccsd.t1_sp
        self.t2_sp = ccsd.t2_sp
        self.ttau = cclambda.build_tau(self.t1_sp, self.t2_sp)
        self.Loovv = cclambda.build_Loovv()
        self.Looov = cclambda.build_Looov()
        self.Lvovv = cclambda.build_Lvovv()
        self.Hov = cclambda.build_Hov(self.Fsp, self.t1_sp)
        self.Hvv = cclambda.build_Hvv(self.Fsp, self.t1_sp, self.t2_sp)
        self.Hoo = cclambda.build_Hoo(self.Fsp, self.t1_sp, self.t2_sp)
        self.Hoooo = cclambda.build_Hoooo(self.t1_sp, self.t2_sp)
        self.Hvvvv = cclambda.build_Hvvvv(self.t1_sp, self.t2_sp)
        self.Hvovv = cclambda.build_Hvovv(self.t1_sp)
        self.Hooov = cclambda.build_Hooov(self.t1_sp)
        self.Hovvo = cclambda.build_Hovvo(self.t1_sp, self.t2_sp)
        self.Hovov = cclambda.build_Hovov(self.t1_sp, self.t2_sp)
        self.Hvvvo = cclambda.build_Hvvvo(self.Fsp, self.t1_sp, self.t2_sp)
        self.Hovoo = cclambda.build_Hovoo(self.Fsp, self.t1_sp, self.t2_sp)
        self.l1 = cclambda.l1
        self.l2 = cclambda.l2
        self.omega = omega

        # dp
        self.Loovv_dp = cclambda.build_Loovv_dp()
        self.Looov_dp = cclambda.build_Looov_dp()
        self.Lvovv_dp = cclambda.build_Lvovv_dp()
        self.Hov_dp = cclambda.build_Hov(self.F, self.t1)
        self.Hvv_dp = cclambda.build_Hvv(self.F, self.t1, self.t2)
        self.Hoo_dp = cclambda.build_Hoo(self.F, self.t1, self.t2)
        self.Hoooo_dp = cclambda.build_Hoooo(self.t1, self.t2)
        self.Hvvvv_dp = cclambda.build_Hvvvv(self.t1, self.t2)
        self.Hvovv_dp = cclambda.build_Hvovv(self.t1)
        self.Hooov_dp = cclambda.build_Hooov(self.t1)
        self.Hovvo_dp = cclambda.build_Hovvo(self.t1, self.t2)
        self.Hovov_dp = cclambda.build_Hovov(self.t1, self.t2)
        self.Hvvvo_dp = cclambda.build_Hvvvo(self.F, self.t1, self.t2)
        self.Hovoo_dp = cclambda.build_Hovoo(self.F, self.t1, self.t2)
        #

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'o': self.slice_o, 'v': self.slice_v,
                           'a': self.slice_a}

        # Build the denominators from diagonal elements of Hbar and omega
        self.Dia = self.Hoo.diagonal().reshape(-1, 1) - self.Hvv.diagonal()
        self.Dijab = self.Hoo.diagonal().reshape(-1, 1, 1, 1) + self.Hoo.diagonal().reshape(-1, 1,
                                                                                            1) - self.Hvv.diagonal().reshape(
            -1, 1) - self.Hvv.diagonal()
        self.Dia += omega
        self.Dijab += omega

        # Guesses for X1 and X2 amplitudes (First order perturbed T amplitudes)
        self.x1 = self.build_Avo_dp().swapaxes(0, 1) / self.Dia
        self.pertbar_ijab = self.build_Avvoo_dp().swapaxes(0, 2).swapaxes(1, 3)
        self.x2 = self.pertbar_ijab.copy()
        self.x2 += self.pertbar_ijab.swapaxes(0, 1).swapaxes(2, 3)
        self.x2 = self.x2 / self.Dijab

        # Guesses for Y1 and Y2 amplitudes (First order perturbed Lambda amplitudes)
        self.y1 = 2.0 * self.x1.copy()
        self.y2 = 4.0 * self.x2.copy()
        self.y2 -= 2.0 * self.x2.swapaxes(2, 3)

        # Single-precision
        self.l1_sp = cclambda.l1_sp
        self.l2_sp = cclambda.l2_sp
        self.x1_sp = np.float32(self.x1)
        self.x2_sp = np.float32(self.x2)
        self.y1_sp = np.float32(self.y1)
        self.y2_sp = np.float32(self.y2)

        # Conventions used :
        # occ orbitals  : i, j, k, l, m, n
        # virt orbitals : a, b, c, d, e, f
        # all oribitals : p, q, r, s, t, u, v

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

    def get_F(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 2 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def get_F_sp(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 2 elements.' % string)
        return self.Fsp[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def get_pert(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_pert: string %s must have 2 elements.' % string)
        return self.pert[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    def get_pert_sp(self, string):
        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_pert: string %s must have 2 elements.' % string)
        return self.pert_sp[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    # Build different pieces of the similarity transformed perturbation operator
    # using ground state T amplitudes i.e T(0).
    # A_bar = e^{-T(0)} A e^{T(0)} = A + [A,T(0)] + 1/2! [[A,T(0)],T(0)]
    # since A is a one body operator, the expansion truncates at double commutators.

    def build_Aoo(self):
        Aoo = self.get_pert_sp('oo').copy()
        Aoo += ndot('ie,me->mi', self.t1_sp, self.get_pert_sp('ov'))
        return Aoo
    def build_Aoo_dp(self):
        Aoo = self.get_pert('oo').copy()
        Aoo += ndot('ie,me->mi', self.t1, self.get_pert('ov'))
        return Aoo

    def build_Aov(self):
        Aov = self.get_pert_sp('ov').copy()
        return Aov
    def build_Aov_dp(self):
        Aov = self.get_pert('ov').copy()
        return Aov

    def build_Avo(self):
        Avo = self.get_pert_sp('vo').copy()
        Avo += ndot('ae,ie->ai', self.get_pert_sp('vv'), self.t1_sp)
        Avo -= ndot('ma,mi->ai', self.t1_sp, self.get_pert_sp('oo'))
        Avo += ndot('miea,me->ai', self.t2_sp, self.get_pert_sp('ov'), prefactor=2.0)
        Avo += ndot('imea,me->ai', self.t2_sp, self.get_pert_sp('ov'), prefactor=-1.0)
        tmp = ndot('ie,ma->imea', self.t1_sp, self.t1_sp)
        Avo -= ndot('imea,me->ai', tmp, self.get_pert_sp('ov'))
        return Avo
    def build_Avo_dp(self):
        Avo = self.get_pert('vo').copy()
        Avo += ndot('ae,ie->ai', self.get_pert('vv'), self.t1)
        Avo -= ndot('ma,mi->ai', self.t1, self.get_pert('oo'))
        Avo += ndot('miea,me->ai', self.t2, self.get_pert('ov'), prefactor=2.0)
        Avo += ndot('imea,me->ai', self.t2, self.get_pert('ov'), prefactor=-1.0)
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        Avo -= ndot('imea,me->ai', tmp, self.get_pert('ov'))
        return Avo

    def build_Avv(self):
        Avv = self.get_pert_sp('vv').copy()
        Avv -= ndot('ma,me->ae', self.t1_sp, self.get_pert_sp('ov'))
        return Avv
    def build_Avv_dp(self):
        Avv = self.get_pert('vv').copy()
        Avv -= ndot('ma,me->ae', self.t1, self.get_pert('ov'))
        return Avv

    def build_Aovoo(self):
        Aovoo = 0
        Aovoo += ndot('ijeb,me->mbij', self.t2_sp, self.get_pert_sp('ov'))
        return Aovoo
    def build_Aovoo_dp(self):
        Aovoo = 0
        Aovoo += ndot('ijeb,me->mbij', self.t2, self.get_pert('ov'))
        return Aovoo

    def build_Avvvo(self):
        Avvvo = 0
        Avvvo -= ndot('miab,me->abei', self.t2_sp, self.get_pert_sp('ov'))
        return Avvvo
    def build_Avvvo_dp(self):
        Avvvo = 0
        Avvvo -= ndot('miab,me->abei', self.t2, self.get_pert('ov'))
        return Avvvo

    def build_Avvoo(self):
        Avvoo = 0
        Avvoo += ndot('ijeb,ae->abij', self.t2_sp, self.build_Avv())
        Avvoo -= ndot('mjab,mi->abij', self.t2_sp, self.build_Aoo())
        return Avvoo
    def build_Avvoo_dp(self):
        Avvoo = 0
        Avvoo += ndot('ijeb,ae->abij', self.t2, self.build_Avv_dp())
        Avvoo -= ndot('mjab,mi->abij', self.t2, self.build_Aoo_dp())
        return Avvoo
    # Intermediates to avoid construction of 3 body Hbar terms
    # in solving X amplitude equations.
    def build_Zvv(self):
        Zvv = 0
        Zvv += ndot('amef,mf->ae', self.Hvovv, self.x1_sp, prefactor=2.0)
        Zvv += ndot('amfe,mf->ae', self.Hvovv, self.x1_sp, prefactor=-1.0)
        Zvv -= ndot('mnaf,mnef->ae', self.x2_sp, self.Loovv)
        return Zvv

    def build_Zoo(self):
        Zoo = 0
        Zoo -= ndot('mnie,ne->mi', self.Hooov, self.x1_sp, prefactor=2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, self.x1_sp, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.Loovv, self.x2_sp)
        return Zoo

    # Intermediates to avoid construction of 3 body Hbar terms
    # in solving Y amplitude equations (just like in lambda equations).
    def build_Goo(self, t2, y2):
        Goo = 0
        Goo += ndot('mjab,ijab->mi', t2, y2)
        return Goo

    def build_Gvv(self, y2, t2):
        Gvv = 0
        Gvv -= ndot('ijab,ijeb->ae', y2, t2)
        return Gvv

    def update_X(self):
        # X1 and X2 amplitudes are the Fourier analogues of first order perturbed T1 and T2 amplitudes,
        # (eq. 65, [Crawford:xxxx]). For a given perturbation, these amplitudes are frequency dependent and
        # can be obtained by solving a linear system of equations, (Hbar(0) - omgea * I)X = Hbar(1)
        # Refer to eq 70 of [Crawford:xxxx]. Writing t_mu^(1)(omega) as X_mu and Hbar^(1)(omega) as A_bar,
        # X1 equations:
        # omega * X_ia = <phi^a_i|A_bar|O> + <phi^a_i|Hbar^(0)|phi^c_k> * X_kc + <phi^a_i|Hbar^(0)|phi^cd_kl> * X_klcd
        # X2 equations:
        # omega * X_ijab = <phi^ab_ij|A_bar|O> + <phi^ab_ij|Hbar^(0)|phi^c_k> * X_kc + <phi^ab_ij|Hbar^(0)|phi^cd_kl> * X_klcd
        # Note that the RHS terms have exactly the same structure as EOM-CCSD sigma equations.
        # Spin Orbital expressions (Einstein summation):

        # X1 equations:
        # -omega * X_ia + A_bar_ai + X_ie * Hvv_ae - X_ma * Hoo_mi + X_me * Hovvo_maei + X_miea * Hov_me
        # + 0.5 * X_imef * Hvovv_amef - 0.5 * X_mnae * Hooov_mnie = 0

        # X2 equations:
        # -omega * X_ijab + A_bar_abij + P(ij) X_ie * Hvvvo_abej - P(ab) X_ma * Hovoo_mbij
        # + P(ab) X_mf * Hvovv_amef * t_ijeb - P(ij) X_ne * Hooov_mnie * t_mjab
        # + P(ab) X_ijeb * Hvv_ae  - P(ij) X_mjab * Hov_mi + 0.5 * X_mnab * Hoooo_mnij + 0.5 * X_ijef * Hvvvv_abef
        # + P(ij) P(ab) X_miea * Hovvo_mbej - 0.5 * P(ab) X_mnaf * Hoovv_mnef * t_ijeb
        # - 0.5 * P(ij) X_inef * Hoovv_mnef * t_mjab

        # It should be noted that in order to avoid construction of 3-body Hbar terms appearing in X2 equations like,
        # Hvvooov_bamjif = Hvovv_amef * t_ijeb,
        # Hvvooov_banjie = Hooov_mnie * t_mjab,
        # Hvoooov_bmnjif = Hoovv_mnef * t_ijeb,
        # Hvvoovv_banjef = Hoovv_mnef * t_mjab,
        # we make use of Z intermediates:
        # Zvv_ae = - Hooov_amef * X_mf - 0.5 * X_mnaf * Hoovv_mnef,
        # Zoo_mi = - X_ne * Hooov_mnie - 0.5 * Hoovv_mnef * X_inef,
        # And then contract Z with T2 amplitudes.

        # X1 equations
        r_x1 = self.build_Avo_dp().swapaxes(0, 1).copy()
        r_x1 -= self.omega * self.x1.copy()
        r_x1 += np.float64(ndot('ie,ae->ia', self.x1_sp, self.Hvv))
        r_x1 -= np.float64(ndot('mi,ma->ia', self.Hoo, self.x1_sp))
        tmp = ndot('maei,me->ia', self.Hovvo, self.x1_sp, prefactor=2.0)
        tmp += ndot('maie,me->ia', self.Hovov, self.x1_sp, prefactor=-1.0)
        r_x1 += np.float64(tmp)
        tmp= ndot('miea,me->ia', self.x2_sp, self.Hov, prefactor=2.0)
        tmp += ndot('imea,me->ia', self.x2_sp, self.Hov, prefactor=-1.0)
        r_x1 += np.float64(tmp)
        tmp = ndot('imef,amef->ia', self.x2_sp, self.Hvovv, prefactor=2.0)
        tmp += ndot('imef,amfe->ia', self.x2_sp, self.Hvovv, prefactor=-1.0)
        r_x1 += np.float64(tmp)
        tmp = ndot('mnie,mnae->ia', self.Hooov, self.x2_sp, prefactor=2.0)
        tmp += ndot('nmie,mnae->ia', self.Hooov, self.x2_sp, prefactor=-1.0)
        r_x1 -= np.float64(tmp)
        # X1 equations over!

        # X2 equations
        # Final r_x2_ijab = r_x2_ijab + r_x2_jiba
        r_x2 = self.build_Avvoo_dp().swapaxes(0, 2).swapaxes(1, 3).copy()
        # a factor of 0.5 because of the comment just above
        # and due to the fact that X2_ijab = X2_jiba
        r_x2 -= 0.5 * self.omega * self.x2
        r_x2 += np.float64(ndot('ie,abej->ijab', self.x1_sp, self.Hvvvo))
        r_x2 -= np.float64(ndot('mbij,ma->ijab', self.Hovoo, self.x1_sp))
        r_x2 += np.float64(ndot('ijeb,ae->ijab', self.x2_sp, self.Hvv))
        r_x2 -= np.float64(ndot('mi,mjab->ijab', self.Hoo, self.x2_sp))
        r_x2 += np.float64(ndot('mnij,mnab->ijab', self.Hoooo, self.x2_sp, prefactor=0.5))
        r_x2 += np.float64(ndot('ijef,abef->ijab', self.x2_sp, self.Hvvvv, prefactor=0.5))
        tmp = ndot('miea,mbej->ijab', self.x2_sp, self.Hovvo, prefactor=2.0)
        tmp += ndot('miea,mbje->ijab', self.x2_sp, self.Hovov, prefactor=-1.0)
        r_x2 += np.float64(tmp)
        r_x2 -= np.float64(ndot('imeb,maje->ijab', self.x2_sp, self.Hovov))
        r_x2 -= np.float64(ndot('imea,mbej->ijab', self.x2_sp, self.Hovvo))
        r_x2 += np.float64(ndot('mi,mjab->ijab', self.build_Zoo(), self.t2_sp))
        r_x2 += np.float64(ndot('ijeb,ae->ijab', self.t2_sp, self.build_Zvv()))
        # X2 equations over!

        old_x2 = self.x2.copy()
        old_x1 = self.x1.copy()

        # update X1 and X2
        self.x1 += r_x1 / self.Dia
        # Final r_x2_ijab = r_x2_ijab + r_x2_jiba
        tmp = r_x2 / self.Dijab
        self.x2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)
        # update single-precision X1, X2
        self.x1_sp = np.float32(self.x1)
        self.x2_sp = np.float32(self.x2)

        # Calcuate rms with the residual
        rms = 0
        rms += np.einsum('ia,ia->', old_x1 - self.x1, old_x1 - self.x1)
        rms += np.einsum('ijab,ijab->', old_x2 - self.x2, old_x2 - self.x2)
        return np.sqrt(rms)

    ### Test if ihomo_y1 ihomo_y2 is in dp
    def inhomogenous_y2_dp(self):

        # Inhomogenous terms appearing in Y2 equations
        # <O|L1(0)|A_bar|phi^ab_ij>
        r_y2 = ndot('ia,jb->ijab', self.l1, self.build_Aov_dp(), prefactor=2.0)
        r_y2 -= ndot('ja,ib->ijab', self.l1, self.build_Aov_dp())
        # <O|L2(0)|A_bar|phi^ab_ij>
        r_y2 += ndot('ijeb,ea->ijab', self.l2, self.build_Avv_dp())
        r_y2 -= ndot('im,mjab->ijab', self.build_Aoo_dp(), self.l2)
        # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij>
        tmp = ndot('me,ja->meja', self.x1, self.l1)
        r_y2 -= ndot('mieb,meja->ijab', self.Loovv_dp, tmp)
        tmp = ndot('me,mb->eb', self.x1, self.l1)
        r_y2 -= ndot('ijae,eb->ijab', self.Loovv_dp, tmp)
        tmp = ndot('me,ie->mi', self.x1, self.l1)
        r_y2 -= ndot('mi,jmba->ijab', tmp, self.Loovv_dp)
        tmp = ndot('me,jb->mejb', self.x1, self.l1, prefactor=2.0)
        r_y2 += ndot('imae,mejb->ijab', self.Loovv_dp, tmp)
        # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij>
        tmp = ndot('me,ma->ea', self.x1, self.Hov_dp)
        r_y2 -= ndot('ijeb,ea->ijab', self.l2, tmp)
        tmp = ndot('me,ie->mi', self.x1, self.Hov_dp)
        r_y2 -= ndot('mi,jmba->ijab', tmp, self.l2)
        tmp = ndot('me,ijef->mijf', self.x1, self.l2)
        r_y2 -= ndot('mijf,fmba->ijab', tmp, self.Hvovv_dp)
        tmp = ndot('me,imbf->eibf', self.x1, self.l2)
        r_y2 -= ndot('eibf,fjea->ijab', tmp, self.Hvovv_dp)
        tmp = ndot('me,jmfa->ejfa', self.x1, self.l2)
        r_y2 -= ndot('fibe,ejfa->ijab', self.Hvovv_dp, tmp)
        tmp = ndot('me,fmae->fa', self.x1, self.Hvovv_dp, prefactor=2.0)
        tmp -= ndot('me,fmea->fa', self.x1, self.Hvovv_dp)
        r_y2 += ndot('ijfb,fa->ijab', self.l2, tmp)
        tmp = ndot('me,fiea->mfia', self.x1, self.Hvovv_dp, prefactor=2.0)
        tmp -= ndot('me,fiae->mfia', self.x1, self.Hvovv_dp)
        r_y2 += ndot('mfia,jmbf->ijab', tmp, self.l2)
        tmp = ndot('me,jmna->ejna', self.x1, self.Hooov_dp)
        r_y2 += ndot('ineb,ejna->ijab', self.l2, tmp)
        tmp = ndot('me,mjna->ejna', self.x1, self.Hooov_dp)
        r_y2 += ndot('nieb,ejna->ijab', self.l2, tmp)
        tmp = ndot('me,nmba->enba', self.x1, self.l2)
        r_y2 += ndot('jine,enba->ijab', self.Hooov_dp, tmp)
        tmp = ndot('me,mina->eina', self.x1, self.Hooov_dp, prefactor=2.0)
        tmp -= ndot('me,imna->eina', self.x1, self.Hooov_dp)
        r_y2 -= ndot('eina,njeb->ijab', tmp, self.l2)
        tmp = ndot('me,imne->in', self.x1, self.Hooov_dp, prefactor=2.0)
        tmp -= ndot('me,mine->in', self.x1, self.Hooov_dp)
        r_y2 -= ndot('in,jnba->ijab', tmp, self.l2)
        # <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>
        tmp = ndot('ijef,mnef->ijmn', self.l2, self.x2, prefactor=0.5)
        r_y2 += ndot('ijmn,mnab->ijab', tmp, self.get_MO('oovv'))
        tmp = ndot('ijfe,mnef->ijmn', self.get_MO('oovv'), self.x2, prefactor=0.5)
        r_y2 += ndot('ijmn,mnba->ijab', tmp, self.l2)
        tmp = ndot('mifb,mnef->ibne', self.l2, self.x2)
        r_y2 += ndot('ibne,jnae->ijab', tmp, self.get_MO('oovv'))
        tmp = ndot('imfb,mnef->ibne', self.l2, self.x2)
        r_y2 += ndot('ibne,njae->ijab', tmp, self.get_MO('oovv'))
        tmp = ndot('mjfb,mnef->jbne', self.l2, self.x2)
        r_y2 -= ndot('jbne,inae->ijab', tmp, self.Loovv_dp)
        r_y2 -= ndot('in,jnba->ijab', self.build_Goo(self.Loovv_dp, self.x2), self.l2)
        r_y2 += ndot('ijfb,af->ijab', self.l2, self.build_Gvv(self.Loovv_dp, self.x2))
        r_y2 += ndot('ijae,be->ijab', self.Loovv_dp, self.build_Gvv(self.l2, self.x2))
        r_y2 -= ndot('imab,jm->ijab', self.Loovv_dp, self.build_Goo(self.l2, self.x2))
        tmp = ndot('nifb,mnef->ibme', self.l2, self.x2)
        r_y2 -= ndot('ibme,mjea->ijab', tmp, self.Loovv_dp)
        tmp = ndot('njfb,mnef->jbme', self.l2, self.x2, prefactor=2.0)
        r_y2 += ndot('imae,jbme->ijab', self.Loovv_dp, tmp)

        return r_y2

    def inhomogenous_y1_dp(self):

        # Inhomogenous terms appearing in Y1 equations
        # <O|A_bar|phi^a_i>
        r_y1 = 2.0 * self.build_Aov_dp().copy()
        # <O|L1(0)|A_bar|phi^a_i>
        r_y1 -= ndot('im,ma->ia', self.build_Aoo_dp(), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Avv_dp())
        # <O|L2(0)|A_bar|phi^a_i>
        r_y1 += ndot('imfe,feam->ia', self.l2, self.build_Avvvo_dp())
        r_y1 -= ndot('ienm,mnea->ia', self.build_Aovoo_dp(), self.l2, prefactor=0.5)
        r_y1 -= ndot('iemn,mnae->ia', self.build_Aovoo_dp(), self.l2, prefactor=0.5)
        # <O|[Hbar(0), X1]|phi^a_i>
        r_y1 += ndot('imae,me->ia', self.Loovv_dp, self.x1, prefactor=2.0)
        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp = ndot('ma,ie->miae', self.Hov_dp, self.l1, prefactor=-1.0)
        tmp -= ndot('ma,ie->miae', self.l1, self.Hov_dp)
        tmp -= ndot('mina,ne->miae', self.Hooov_dp, self.l1, prefactor=2.0)
        tmp -= ndot('imna,ne->miae', self.Hooov_dp, self.l1, prefactor=-1.0)
        tmp -= ndot('imne,na->miae', self.Hooov_dp, self.l1, prefactor=2.0)
        tmp -= ndot('mine,na->miae', self.Hooov_dp, self.l1, prefactor=-1.0)
        tmp += ndot('fmae,if->miae', self.Hvovv_dp, self.l1, prefactor=2.0)
        tmp += ndot('fmea,if->miae', self.Hvovv_dp, self.l1, prefactor=-1.0)
        tmp += ndot('fiea,mf->miae', self.Hvovv_dp, self.l1, prefactor=2.0)
        tmp += ndot('fiae,mf->miae', self.Hvovv_dp, self.l1, prefactor=-1.0)
        r_y1 += ndot('miae,me->ia', tmp, self.x1)
        # <O|L1(0)|[Hbar(0), X2]|phi^a_i>
        tmp = ndot('mnef,nf->me', self.x2, self.l1, prefactor=2.0)
        tmp += ndot('mnfe,nf->me', self.x2, self.l1, prefactor=-1.0)
        r_y1 += ndot('imae,me->ia', self.Loovv_dp, tmp)
        r_y1 -= ndot('ni,na->ia', self.build_Goo(self.x2, self.Loovv_dp), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Gvv(self.x2, self.Loovv_dp))
        # <O|L2(0)|[Hbar(0), X1]|phi^a_i>
        tmp = ndot('nief,mfna->iema', self.l2, self.Hovov_dp, prefactor=-1.0)
        tmp -= ndot('ifne,nmaf->iema', self.Hovov_dp, self.l2)
        tmp -= ndot('inef,mfan->iema', self.l2, self.Hovvo_dp)
        tmp -= ndot('ifen,nmfa->iema', self.Hovvo_dp, self.l2)
        tmp += ndot('imfg,fgae->iema', self.l2, self.Hvvvv_dp, prefactor=0.5)
        tmp += ndot('imgf,fgea->iema', self.l2, self.Hvvvv_dp, prefactor=0.5)
        tmp += ndot('imno,onea->iema', self.Hoooo_dp, self.l2, prefactor=0.5)
        tmp += ndot('mino,noea->iema', self.Hoooo_dp, self.l2, prefactor=0.5)
        r_y1 += ndot('iema,me->ia', tmp, self.x1)
        tmp = ndot('nb,fb->nf', self.x1, self.build_Gvv(self.t2, self.l2))
        r_y1 += ndot('inaf,nf->ia', self.Loovv_dp, tmp)
        tmp = ndot('me,fa->mefa', self.x1, self.build_Gvv(self.t2, self.l2))
        r_y1 += ndot('mief,mefa->ia', self.Loovv_dp, tmp)
        tmp = ndot('me,ni->meni', self.x1, self.build_Goo(self.t2, self.l2))
        r_y1 -= ndot('meni,mnea->ia', tmp, self.Loovv_dp)
        tmp = ndot('jf,nj->fn', self.x1, self.build_Goo(self.t2, self.l2))
        r_y1 -= ndot('inaf,fn->ia', self.Loovv_dp, tmp)
        # <O|L2(0)|[Hbar(0), X2]|phi^a_i>
        r_y1 -= ndot('mi,ma->ia', self.build_Goo(self.x2, self.l2), self.Hov_dp)
        r_y1 += ndot('ie,ea->ia', self.Hov_dp, self.build_Gvv(self.x2, self.l2))
        tmp = ndot('imfg,mnef->igne', self.l2, self.x2)
        r_y1 -= ndot('igne,gnea->ia', tmp, self.Hvovv_dp)
        tmp = ndot('mifg,mnef->igne', self.l2, self.x2)
        r_y1 -= ndot('igne,gnae->ia', tmp, self.Hvovv_dp)
        tmp = ndot('mnga,mnef->gaef', self.l2, self.x2)
        r_y1 -= ndot('gief,gaef->ia', self.Hvovv_dp, tmp)
        tmp = ndot('gmae,mnef->ganf', self.Hvovv_dp, self.x2, prefactor=2.0)
        tmp += ndot('gmea,mnef->ganf', self.Hvovv_dp, self.x2, prefactor=-1.0)
        r_y1 += ndot('nifg,ganf->ia', self.l2, tmp)
        r_y1 -= ndot('giea,ge->ia', self.Hvovv_dp, self.build_Gvv(self.l2, self.x2), prefactor=2.0)
        r_y1 -= ndot('giae,ge->ia', self.Hvovv_dp, self.build_Gvv(self.l2, self.x2), prefactor=-1.0)
        tmp = ndot('oief,mnef->oimn', self.l2, self.x2)
        r_y1 += ndot('oimn,mnoa->ia', tmp, self.Hooov_dp)
        tmp = ndot('mofa,mnef->oane', self.l2, self.x2)
        r_y1 += ndot('inoe,oane->ia', self.Hooov_dp, tmp)
        tmp = ndot('onea,mnef->oamf', self.l2, self.x2)
        r_y1 += ndot('miof,oamf->ia', self.Hooov_dp, tmp)
        r_y1 -= ndot('mioa,mo->ia', self.Hooov_dp, self.build_Goo(self.x2, self.l2), prefactor=2.0)
        r_y1 -= ndot('imoa,mo->ia', self.Hooov_dp, self.build_Goo(self.x2, self.l2), prefactor=-1.0)
        tmp = ndot('imoe,mnef->ionf', self.Hooov_dp, self.x2, prefactor=-2.0)
        tmp -= ndot('mioe,mnef->ionf', self.Hooov_dp, self.x2, prefactor=-1.0)
        r_y1 += ndot('ionf,nofa->ia', tmp, self.l2)

        return r_y1

    def inhomogenous_y2(self):

        # Inhomogenous terms appearing in Y2 equations
        # <O|L1(0)|A_bar|phi^ab_ij>
        tmp = ndot('ia,jb->ijab', self.l1_sp, self.build_Aov(), prefactor=2.0)
        tmp -= ndot('ja,ib->ijab', self.l1_sp, self.build_Aov())
        r_y2 = np.float64(tmp)
        # <O|L2(0)|A_bar|phi^ab_ij>
        tmp = ndot('ijeb,ea->ijab', self.l2_sp, self.build_Avv())
        tmp -= ndot('im,mjab->ijab', self.build_Aoo(), self.l2_sp)
        r_y2 += np.float64(tmp)
        # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij>
        tmp = ndot('me,ja->meja', self.x1_sp, self.l1_sp)
        tmp_y2 = ndot('mieb,meja->ijab', self.Loovv, tmp)
        tmp = ndot('me,mb->eb', self.x1_sp, self.l1_sp)
        tmp_y2 += ndot('ijae,eb->ijab', self.Loovv, tmp)
        tmp = ndot('me,ie->mi', self.x1_sp, self.l1_sp)
        tmp_y2 += ndot('mi,jmba->ijab', tmp, self.Loovv)
        tmp = ndot('me,jb->mejb', self.x1_sp, self.l1_sp, prefactor=2.0)
        tmp_y2 -= ndot('imae,mejb->ijab', self.Loovv, tmp)
        r_y2 -= np.float64(tmp_y2)
        # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij>
        tmp = ndot('me,ma->ea', self.x1_sp, self.Hov)
        tmp_y2 = ndot('ijeb,ea->ijab', self.l2_sp, tmp)
        tmp = ndot('me,ie->mi', self.x1_sp, self.Hov)
        tmp_y2 += ndot('mi,jmba->ijab', tmp, self.l2_sp)
        tmp = ndot('me,ijef->mijf', self.x1_sp, self.l2_sp)
        tmp_y2 += ndot('mijf,fmba->ijab', tmp, self.Hvovv)
        tmp = ndot('me,imbf->eibf', self.x1_sp, self.l2_sp)
        tmp_y2 += ndot('eibf,fjea->ijab', tmp, self.Hvovv)
        tmp = ndot('me,jmfa->ejfa', self.x1_sp, self.l2_sp)
        tmp_y2 += ndot('fibe,ejfa->ijab', self.Hvovv, tmp)
        tmp = ndot('me,fmae->fa', self.x1_sp, self.Hvovv, prefactor=2.0)
        tmp -= ndot('me,fmea->fa', self.x1_sp, self.Hvovv)
        tmp_y2 -= ndot('ijfb,fa->ijab', self.l2_sp, tmp)
        tmp = ndot('me,fiea->mfia', self.x1_sp, self.Hvovv, prefactor=2.0)
        tmp -= ndot('me,fiae->mfia', self.x1_sp, self.Hvovv)
        tmp_y2 -= ndot('mfia,jmbf->ijab', tmp, self.l2_sp)
        tmp = ndot('me,jmna->ejna', self.x1_sp, self.Hooov)
        tmp_y2 -= ndot('ineb,ejna->ijab', self.l2_sp, tmp)
        tmp = ndot('me,mjna->ejna', self.x1_sp, self.Hooov)
        tmp_y2 -= ndot('nieb,ejna->ijab', self.l2_sp, tmp)
        tmp = ndot('me,nmba->enba', self.x1_sp, self.l2_sp)
        tmp_y2 -= ndot('jine,enba->ijab', self.Hooov, tmp)
        tmp = ndot('me,mina->eina', self.x1_sp, self.Hooov, prefactor=2.0)
        tmp -= ndot('me,imna->eina', self.x1_sp, self.Hooov)
        tmp_y2 += ndot('eina,njeb->ijab', tmp, self.l2_sp)
        tmp = ndot('me,imne->in', self.x1_sp, self.Hooov, prefactor=2.0)
        tmp -= ndot('me,mine->in', self.x1_sp, self.Hooov)
        tmp_y2 += ndot('in,jnba->ijab', tmp, self.l2_sp)
        r_y2 -= np.float64(tmp_y2)
        # <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>
        tmp = ndot('ijef,mnef->ijmn', self.l2_sp, self.x2_sp, prefactor=0.5)
        tmp_y2 = ndot('ijmn,mnab->ijab', tmp, self.get_MO_sp('oovv'))
        tmp = ndot('ijfe,mnef->ijmn', self.get_MO_sp('oovv'), self.x2_sp, prefactor=0.5)
        tmp_y2 += ndot('ijmn,mnba->ijab', tmp, self.l2_sp)
        tmp = ndot('mifb,mnef->ibne', self.l2_sp, self.x2_sp)
        tmp_y2 += ndot('ibne,jnae->ijab', tmp, self.get_MO_sp('oovv'))
        tmp = ndot('imfb,mnef->ibne', self.l2_sp, self.x2_sp)
        tmp_y2 += ndot('ibne,njae->ijab', tmp, self.get_MO_sp('oovv'))
        tmp = ndot('mjfb,mnef->jbne', self.l2_sp, self.x2_sp)
        tmp_y2 -= ndot('jbne,inae->ijab', tmp, self.Loovv)
        tmp_y2 -= ndot('in,jnba->ijab', self.build_Goo(self.Loovv, self.x2_sp), self.l2_sp)
        tmp_y2 += ndot('ijfb,af->ijab', self.l2_sp, self.build_Gvv(self.Loovv, self.x2_sp))
        tmp_y2 += ndot('ijae,be->ijab', self.Loovv, self.build_Gvv(self.l2_sp, self.x2_sp))
        tmp_y2 -= ndot('imab,jm->ijab', self.Loovv, self.build_Goo(self.l2_sp, self.x2_sp))
        tmp = ndot('nifb,mnef->ibme', self.l2_sp, self.x2_sp)
        tmp_y2 -= ndot('ibme,mjea->ijab', tmp, self.Loovv)
        tmp = ndot('njfb,mnef->jbme', self.l2_sp, self.x2_sp, prefactor=2.0)
        tmp_y2 += ndot('imae,jbme->ijab', self.Loovv, tmp)
        r_y2 += np.float64(tmp_y2)

        return r_y2

    def inhomogenous_y1(self):

        # Inhomogenous terms appearing in Y1 equations
        # <O|A_bar|phi^a_i>
        r_y1 = 2.0 * self.build_Aov_dp().copy()
        # <O|L1(0)|A_bar|phi^a_i>
        tmp = ndot('im,ma->ia', self.build_Aoo(), self.l1_sp)
        tmp -= ndot('ie,ea->ia', self.l1_sp, self.build_Avv())
        r_y1 -= np.float64(tmp)
        # <O|L2(0)|A_bar|phi^a_i>
        tmp = ndot('imfe,feam->ia', self.l2_sp, self.build_Avvvo())
        tmp -= ndot('ienm,mnea->ia', self.build_Aovoo(), self.l2_sp, prefactor=0.5)
        tmp -= ndot('iemn,mnae->ia', self.build_Aovoo(), self.l2_sp, prefactor=0.5)
        r_y1 += np.float64(tmp)
        # <O|[Hbar(0), X1]|phi^a_i>
        r_y1 += np.float64(ndot('imae,me->ia', self.Loovv, self.x1_sp, prefactor=2.0))
        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp = ndot('ma,ie->miae', self.Hov, self.l1_sp, prefactor=-1.0)
        tmp -= ndot('ma,ie->miae', self.l1_sp, self.Hov)
        tmp -= ndot('mina,ne->miae', self.Hooov, self.l1_sp, prefactor=2.0)
        tmp -= ndot('imna,ne->miae', self.Hooov, self.l1_sp, prefactor=-1.0)
        tmp -= ndot('imne,na->miae', self.Hooov, self.l1_sp, prefactor=2.0)
        tmp -= ndot('mine,na->miae', self.Hooov, self.l1_sp, prefactor=-1.0)
        tmp += ndot('fmae,if->miae', self.Hvovv, self.l1_sp, prefactor=2.0)
        tmp += ndot('fmea,if->miae', self.Hvovv, self.l1_sp, prefactor=-1.0)
        tmp += ndot('fiea,mf->miae', self.Hvovv, self.l1_sp, prefactor=2.0)
        tmp += ndot('fiae,mf->miae', self.Hvovv, self.l1_sp, prefactor=-1.0)
        r_y1 += np.float64(ndot('miae,me->ia', tmp, self.x1_sp))
        # <O|L1(0)|[Hbar(0), X2]|phi^a_i>
        tmp = ndot('mnef,nf->me', self.x2_sp, self.l1_sp, prefactor=2.0)
        tmp += ndot('mnfe,nf->me', self.x2_sp, self.l1_sp, prefactor=-1.0)
        tmp_y1 = ndot('imae,me->ia', self.Loovv, tmp)
        tmp_y1 -= ndot('ni,na->ia', self.build_Goo(self.x2_sp, self.Loovv), self.l1_sp)
        tmp_y1 += ndot('ie,ea->ia', self.l1_sp, self.build_Gvv(self.x2_sp, self.Loovv))
        r_y1 += np.float64(tmp_y1)
        # <O|L2(0)|[Hbar(0), X1]|phi^a_i>
        tmp = ndot('nief,mfna->iema', self.l2_sp, self.Hovov, prefactor=-1.0)
        tmp -= ndot('ifne,nmaf->iema', self.Hovov, self.l2_sp)
        tmp -= ndot('inef,mfan->iema', self.l2_sp, self.Hovvo)
        tmp -= ndot('ifen,nmfa->iema', self.Hovvo, self.l2_sp)
        tmp += ndot('imfg,fgae->iema', self.l2_sp, self.Hvvvv, prefactor=0.5)
        tmp += ndot('imgf,fgea->iema', self.l2_sp, self.Hvvvv, prefactor=0.5)
        tmp += ndot('imno,onea->iema', self.Hoooo, self.l2_sp, prefactor=0.5)
        tmp += ndot('mino,noea->iema', self.Hoooo, self.l2_sp, prefactor=0.5)
        tmp_y1 = ndot('iema,me->ia', tmp, self.x1_sp)
        tmp = ndot('nb,fb->nf', self.x1_sp, self.build_Gvv(self.t2_sp, self.l2_sp))
        tmp_y1 += ndot('inaf,nf->ia', self.Loovv, tmp)
        tmp = ndot('me,fa->mefa', self.x1_sp, self.build_Gvv(self.t2_sp, self.l2_sp))
        tmp_y1 += ndot('mief,mefa->ia', self.Loovv, tmp)
        tmp = ndot('me,ni->meni', self.x1_sp, self.build_Goo(self.t2_sp, self.l2_sp))
        tmp_y1 -= ndot('meni,mnea->ia', tmp, self.Loovv)
        tmp = ndot('jf,nj->fn', self.x1_sp, self.build_Goo(self.t2_sp, self.l2_sp))
        tmp_y1 -= ndot('inaf,fn->ia', self.Loovv, tmp)
        r_y1 += np.float64(tmp_y1)
        # <O|L2(0)|[Hbar(0), X2]|phi^a_i>
        tmp_y1 = ndot('mi,ma->ia', self.build_Goo(self.x2_sp, self.l2_sp), self.Hov)
        tmp_y1 -= ndot('ie,ea->ia', self.Hov, self.build_Gvv(self.x2_sp, self.l2_sp))
        tmp = ndot('imfg,mnef->igne', self.l2_sp, self.x2_sp)
        tmp_y1 += ndot('igne,gnea->ia', tmp, self.Hvovv)
        tmp = ndot('mifg,mnef->igne', self.l2_sp, self.x2_sp)
        tmp_y1 += ndot('igne,gnae->ia', tmp, self.Hvovv)
        tmp = ndot('mnga,mnef->gaef', self.l2_sp, self.x2_sp)
        tmp_y1 += ndot('gief,gaef->ia', self.Hvovv, tmp)
        tmp = ndot('gmae,mnef->ganf', self.Hvovv, self.x2_sp, prefactor=2.0)
        tmp += ndot('gmea,mnef->ganf', self.Hvovv, self.x2_sp, prefactor=-1.0)
        tmp_y1 -= ndot('nifg,ganf->ia', self.l2_sp, tmp)
        tmp_y1 += ndot('giea,ge->ia', self.Hvovv, self.build_Gvv(self.l2_sp, self.x2_sp), prefactor=2.0)
        tmp_y1 += ndot('giae,ge->ia', self.Hvovv, self.build_Gvv(self.l2_sp, self.x2_sp), prefactor=-1.0)
        tmp = ndot('oief,mnef->oimn', self.l2_sp, self.x2_sp)
        tmp_y1 -= ndot('oimn,mnoa->ia', tmp, self.Hooov)
        tmp = ndot('mofa,mnef->oane', self.l2_sp, self.x2_sp)
        tmp_y1 -= ndot('inoe,oane->ia', self.Hooov, tmp)
        tmp = ndot('onea,mnef->oamf', self.l2_sp, self.x2_sp)
        tmp_y1 -= ndot('miof,oamf->ia', self.Hooov, tmp)
        tmp_y1 += ndot('mioa,mo->ia', self.Hooov, self.build_Goo(self.x2_sp, self.l2_sp), prefactor=2.0)
        tmp_y1 += ndot('imoa,mo->ia', self.Hooov, self.build_Goo(self.x2_sp, self.l2_sp), prefactor=-1.0)
        tmp = ndot('imoe,mnef->ionf', self.Hooov, self.x2_sp, prefactor=-2.0)
        tmp -= ndot('mioe,mnef->ionf', self.Hooov, self.x2_sp, prefactor=-1.0)
        tmp_y1 -= ndot('ionf,nofa->ia', tmp, self.l2_sp)
        r_y1 -= np.float64(tmp_y1)

        return r_y1

    def update_Y(self):

        # Y1 and Y2 amplitudes are the Fourier analogues of first order perturbed L1 and L2 amplitudes,
        # While X amplitudes are referred to as right hand perturbed amplitudes, Y amplitudes are the
        # left hand perturbed amplitudes. Just like X1 and X2, they can be obtained by solving a linear
        # sytem of equations. Refer to eq 73 of [Crawford:xxxx]. for Writing l_mu^(1)(omega) as Y_mu,
        # Y1 equations:
        # omega * Y_ia + Y_kc * <phi^c_k|Hbar(0)|phi^a_i>  + Y_klcd * <phi^cd_kl|Hbar(0)|phi^a_i>
        # + <O|(1 + L(0))|Hbar_bar(1)(omega)|phi^a_i> = 0
        # Y2 equations:
        # omega * Y_ijab + Y_kc * <phi^c_k|Hbar(0)|phi^ab_ij>  + Y_klcd * <phi^cd_kl|Hbar(0)|phi^ab_ij>
        # + <O|(1 + L(0))|Hbar_bar(1)(omega)|phi^ab_ij> = 0
        # where Hbar_bar(1)(omega) = Hbar(1) + [Hbar(0), T(1)] = A_bar + [Hbar(0), X]
        # Note that the homogenous terms of Y1 and Y2 equations except the omega term are exactly identical in
        # structure to the L1 and L2 equations and just like lambdas, the equations for these Y amplitudes have
        # been derived using the unitray group approach. Please refer to helper_cclambda file for a complete
        # decsription.

        # Y1 equations
        # Inhomogenous terms
        r_y1 = self.im_y1.copy()
        # Homogenous terms now!
        r_y1 += self.omega * self.y1
        r_y1 += np.float64(ndot('ie,ea->ia', self.y1_sp, self.Hvv))
        r_y1 -= np.float64(ndot('im,ma->ia', self.Hoo, self.y1_sp))
        tmp = ndot('ieam,me->ia', self.Hovvo, self.y1_sp, prefactor=2.0)
        tmp += ndot('iema,me->ia', self.Hovov, self.y1_sp, prefactor=-1.0)
        r_y1 += np.float64(tmp)
        r_y1 += np.float64(ndot('imef,efam->ia', self.y2_sp, self.Hvvvo))
        r_y1 -= np.float64(ndot('iemn,mnae->ia', self.Hovoo, self.y2_sp))
        tmp = ndot('eifa,ef->ia', self.Hvovv, self.build_Gvv(self.y2_sp, self.t2_sp), prefactor=2.0)
        tmp += ndot('eiaf,ef->ia', self.Hvovv, self.build_Gvv(self.y2_sp, self.t2_sp), prefactor=-1.0)
        r_y1 -= np.float64(tmp)
        tmp = ndot('mina,mn->ia', self.Hooov, self.build_Goo(self.t2_sp, self.y2_sp), prefactor=2.0)
        tmp += ndot('imna,mn->ia', self.Hooov, self.build_Goo(self.t2_sp, self.y2_sp), prefactor=-1.0)
        r_y1 -= np.float64(tmp)
        # Y1 equations over!

        # Y2 equations
        # Final r_y2_ijab = r_y2_ijab + r_y2_jiba
        # Inhomogenous terms
        r_y2 = self.im_y2.copy()
        # Homogenous terms now!
        # a factor of 0.5 because of the relation/comment just above
        # and due to the fact that Y2_ijab = Y2_jiba
        r_y2 += 0.5 * self.omega * self.y2.copy()
        tmp = ndot('ia,jb->ijab', self.y1_sp, self.Hov, prefactor=2.0)
        tmp -= ndot('ja,ib->ijab', self.y1_sp, self.Hov)
        r_y2 += np.float64(tmp)
        r_y2 += np.float64(ndot('ijeb,ea->ijab', self.y2_sp, self.Hvv))
        r_y2 -= np.float64(ndot('im,mjab->ijab', self.Hoo, self.y2_sp))
        r_y2 += np.float64(ndot('ijmn,mnab->ijab', self.Hoooo, self.y2_sp, prefactor=0.5))
        r_y2 += np.float64(ndot('ijef,efab->ijab', self.y2_sp, self.Hvvvv, prefactor=0.5))
        tmp = ndot('ie,ejab->ijab', self.y1_sp, self.Hvovv, prefactor=2.0)
        tmp += ndot('ie,ejba->ijab', self.y1_sp, self.Hvovv, prefactor=-1.0)
        r_y2 += np.float64(tmp)
        tmp = ndot('mb,jima->ijab', self.y1_sp, self.Hooov, prefactor=2.0)
        tmp += ndot('mb,ijma->ijab', self.y1_sp, self.Hooov, prefactor=-1.0)
        r_y2 -= np.float64(tmp)
        tmp = ndot('ieam,mjeb->ijab', self.Hovvo, self.y2_sp, prefactor=2.0)
        tmp += ndot('iema,mjeb->ijab', self.Hovov, self.y2_sp, prefactor=-1.0)
        r_y2 += np.float64(tmp)
        r_y2 -= np.float64(ndot('mibe,jema->ijab', self.y2_sp, self.Hovov))
        r_y2 -= np.float64(ndot('mieb,jeam->ijab', self.y2_sp, self.Hovvo))
        r_y2 += np.float64(ndot('ijeb,ae->ijab', self.Loovv, self.build_Gvv(self.y2_sp, self.t2_sp)))
        r_y2 -= np.float64(ndot('mi,mjab->ijab', self.build_Goo(self.t2_sp, self.y2_sp), self.Loovv))
        # Y2 equations over!

        old_y1 = self.y1.copy()
        old_y2 = self.y2.copy()

        # update Y1 and Y2
        self.y1 += r_y1 / self.Dia
        # Final r_y2_ijab = r_y2_ijab + r_y2_jiba
        tmp = r_y2 / self.Dijab
        self.y2 += tmp + tmp.swapaxes(0, 1).swapaxes(2, 3)
        # update Y1, Y2 in single-precision
        self.y1_sp = np.float32(self.y1)
        self.y2_sp = np.float32(self.y2)

        # Calcuate rms from the residual
        rms = np.einsum('ia,ia->', old_y1 - self.y1, old_y1 - self.y1)
        rms += np.einsum('ijab,ijab->', old_y2 - self.y2, old_y2 - self.y2)
        return np.sqrt(rms)

    def pseudoresponse(self, hand):
        polar1 = 0
        polar2 = 0
        if hand == 'right':
            z1 = self.x1;
            z2 = self.x2
        else:
            z1 = self.y1;
            z2 = self.y2

        # To match the pseudoresponse values with PSI4
        polar1 += ndot('ia,ai->', z1, self.build_Avo_dp(), prefactor=2.0)
        tmp = self.pertbar_ijab + self.pertbar_ijab.swapaxes(0, 1).swapaxes(2, 3)
        polar2 += ndot('ijab,ijab->', z2, tmp, prefactor=2.0)
        polar2 += ndot('ijba,ijab->', z2, tmp, prefactor=-1.0)

        return -2.0 * (polar1 + polar2)

    def solve(self, hand, r_conv=1.e-7, maxiter=100, max_diis=8, start_diis=500):

        ### Start of the solve routine
        ccpert_tstart = time.time()

        # calculate the pseudoresponse from guess amplitudes
        pseudoresponse_old = self.pseudoresponse(hand)
        print("CCPERT_%s Iteration %3d: pseudoresponse = %.15f   dE = % .5E " % (
        self.name, 0, pseudoresponse_old, -pseudoresponse_old))

        # Set up DIIS before iterations begin
        if hand == 'right':
            diis_object = helper_diis(self.x1, self.x2, max_diis)
        else:
            diis_object = helper_diis(self.y1, self.y2, max_diis)
            # calculate the inhomogenous terms of the left hand amplitudes equation before iterations begin
            self.im_y1 = self.inhomogenous_y1()
            self.im_y2 = self.inhomogenous_y2()

        # Iterate!
        for CCPERT_iter in range(1, maxiter + 1):

            # Residual build and update
            if hand == 'right':
                rms = self.update_X()
            else:
                rms = self.update_Y()

            # pseudoresponse with updated amplitudes
            pseudoresponse = self.pseudoresponse(hand)
            print(rms)

            # Print CCPERT iteration information
            print('CCPERT_%s Iteration %3d: pseudoresponse = %.15f   dE = % .5E   DIIS = %d' % (
            self.name, CCPERT_iter, pseudoresponse, (pseudoresponse - pseudoresponse_old), diis_object.diis_size))

            # Check convergence
            if (rms < r_conv):
                print('\nCCPERT_%s has converged in %.3f seconds!' % (self.name, time.time() - ccpert_tstart))
                return pseudoresponse

            # Update old pseudoresponse
            pseudoresponse_old = pseudoresponse

            #  Add the new error vector
            if hand == 'right':
                diis_object.add_error_vector(self.x1, self.x2)
            else:
                diis_object.add_error_vector(self.y1, self.y2)

            if CCPERT_iter >= start_diis:
                if hand == 'right':
                    self.x1, self.x2 = diis_object.extrapolate(self.x1, self.x2)
                    self.x1_sp = np.float32(self.x1)
                    self.x2_sp = np.float32(self.x2)
                else:
                    self.y1, self.y2 = diis_object.extrapolate(self.y1, self.y2)
                    self.y1_sp = np.float32(self.y1)
                    self.y2_sp = np.float32(self.y2)

# End HelperCCPert class

class CCLinresp(object):

    def __init__(self, cclambda, ccpert_A, ccpert_B):
        # start of the cclinresp class
        time_init = time.time()
        # Grab all the info from ccpert obejct in single-precision, a and b here are the two
        # perturbations Ex. for dipole polarizabilities, A = mu, B = mu (dipole operator)
        self.ccpert_A = ccpert_A
        self.ccpert_B = ccpert_B
        self.pert_A = ccpert_A.pert
        self.pert_B = ccpert_B.pert
        self.l1 = cclambda.l1
        self.l2 = cclambda.l2
        self.l1_sp = cclambda.l1_sp
        self.l2_sp = cclambda.l2_sp
        # Grab X and Y amplitudes corresponding to perturbation A
        self.x1_A = ccpert_A.x1
        self.x2_A = ccpert_A.x2
        self.y1_A = ccpert_A.y1
        self.y2_A = ccpert_A.y2
        # Grab X and Y amplitudes corresponding to perturbation B
        self.x1_B = ccpert_B.x1
        self.x2_B = ccpert_B.x2
        self.y1_B = ccpert_B.y1
        self.y2_B = ccpert_B.y2
        self.x1_B_sp = ccpert_B.x1_sp
        self.x2_B_sp = ccpert_B.x2_sp
        self.y1_B_sp = ccpert_B.y1_sp
        self.y2_B_sp = ccpert_B.y2_sp

    def linresp(self):
        # Please refer to equation 78 of [Crawford:xxxx].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = Y
        # <<A;B>> =  <0|Y(B) * A_bar|0> + <0|(1+L(0))[A_bar, X(B)]|0>
        #                polar1                    polar2
        self.polar1 = 0
        self.polar2 = 0
        # <0|Y1(B) * A_bar|0>
        self.polar1 += np.float64(ndot("ai,ia->", self.ccpert_A.build_Avo(), self.y1_B_sp))
        # <0|Y2(B) * A_bar|0>
        tmp = ndot("abij,ijab->", self.ccpert_A.build_Avvoo(), self.y2_B_sp, prefactor=0.5)
        tmp += ndot("baji,ijab->", self.ccpert_A.build_Avvoo(), self.y2_B_sp, prefactor=0.5)
        self.polar1 += np.float64(tmp)
        # <0|[A_bar, X(B)]|0>
        self.polar2 += np.float64(ndot("ia,ia->", self.ccpert_A.build_Aov(), self.x1_B_sp, prefactor=2.0))
        # <0|L1(0)[A_bar, X1(B)]|0>
        tmp = ndot('ia,ic->ac', self.l1_sp, self.x1_B_sp)
        tmp_polar2 = ndot('ac,ac->', tmp, self.ccpert_A.build_Avv())
        tmp = ndot('ia,ka->ik', self.l1_sp, self.x1_B_sp)
        tmp_polar2 -= ndot('ik,ki->', tmp, self.ccpert_A.build_Aoo())
        self.polar2 += np.float64(tmp_polar2)
        # <0|L1(0)[A_bar, X2(B)]|0>
        tmp = ndot('ia,jb->ijab', self.l1_sp, self.ccpert_A.build_Aov())
        tmp_polar2 = ndot('ijab,ijab->', tmp, self.x2_B_sp, prefactor=2.0)
        tmp_polar2 += ndot('ijab,ijba->', tmp, self.x2_B_sp, prefactor=-1.0)
        self.polar2 += np.float64(tmp_polar2)

        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = ndot('ijbc,bcaj->ia', self.l2_sp, self.ccpert_A.build_Avvvo())
        tmp_polar2 = ndot('ia,ia->', tmp, self.x1_B_sp)
        tmp = ndot('ijab,kbij->ak', self.l2_sp, self.ccpert_A.build_Aovoo())
        tmp_polar2 -= ndot('ak,ka->', tmp, self.x1_B_sp, prefactor=0.5)
        tmp = ndot('ijab,kaji->bk', self.l2_sp, self.ccpert_A.build_Aovoo())
        tmp_polar2 -= ndot('bk,kb->', tmp, self.x1_B_sp, prefactor=0.5)
        self.polar2 += np.float64(tmp_polar2)

        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = ndot('ijab,kjab->ik', self.l2_sp, self.x2_B_sp)
        tmp_polar2 = ndot('ik,ki->', tmp, self.ccpert_A.build_Aoo(), prefactor=0.5)
        tmp = ndot('ijab,kiba->jk', self.l2_sp, self.x2_B_sp, )
        tmp_polar2 += ndot('jk,kj->', tmp, self.ccpert_A.build_Aoo(), prefactor=0.5)
        tmp = ndot('ijab,ijac->bc', self.l2_sp, self.x2_B_sp, )
        tmp_polar2 -= ndot('bc,bc->', tmp, self.ccpert_A.build_Avv(), prefactor=0.5)
        tmp = ndot('ijab,ijcb->ac', self.l2_sp, self.x2_B_sp, )
        tmp_polar2 -= ndot('ac,ac->', tmp, self.ccpert_A.build_Avv(), prefactor=0.5)
        self.polar2 -= np.float64(tmp_polar2)

        return -1.0 * (self.polar1 + self.polar2)

    def linresp_dp(self):
        # Please refer to equation 78 of [Crawford:xxxx].
        # Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = Y
        # <<A;B>> =  <0|Y(B) * A_bar|0> + <0|(1+L(0))[A_bar, X(B)]|0>
        #                polar1                    polar2
        self.polar1 = 0
        self.polar2 = 0
        # <0|Y1(B) * A_bar|0>
        self.polar1 += ndot("ai,ia->", self.ccpert_A.build_Avo_dp(), self.y1_B)
        # <0|Y2(B) * A_bar|0>
        self.polar1 += ndot("abij,ijab->", self.ccpert_A.build_Avvoo_dp(), self.y2_B, prefactor=0.5)
        self.polar1 += ndot("baji,ijab->", self.ccpert_A.build_Avvoo_dp(), self.y2_B, prefactor=0.5)
        # <0|[A_bar, X(B)]|0>
        self.polar2 += ndot("ia,ia->", self.ccpert_A.build_Aov_dp(), self.x1_B, prefactor=2.0)
        # <0|L1(0)[A_bar, X1(B)]|0>
        tmp = ndot('ia,ic->ac', self.l1, self.x1_B)
        self.polar2 += ndot('ac,ac->', tmp, self.ccpert_A.build_Avv_dp())
        tmp = ndot('ia,ka->ik', self.l1, self.x1_B)
        self.polar2 -= ndot('ik,ki->', tmp, self.ccpert_A.build_Aoo_dp())
        # <0|L1(0)[A_bar, X2(B)]|0>
        tmp = ndot('ia,jb->ijab', self.l1, self.ccpert_A.build_Aov_dp())
        self.polar2 += ndot('ijab,ijab->', tmp, self.x2_B, prefactor=2.0)
        self.polar2 += ndot('ijab,ijba->', tmp, self.x2_B, prefactor=-1.0)
        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = ndot('ijbc,bcaj->ia', self.l2, self.ccpert_A.build_Avvvo_dp())
        self.polar2 += ndot('ia,ia->', tmp, self.x1_B)
        tmp = ndot('ijab,kbij->ak', self.l2, self.ccpert_A.build_Aovoo_dp())
        self.polar2 -= ndot('ak,ka->', tmp, self.x1_B, prefactor=0.5)
        tmp = ndot('ijab,kaji->bk', self.l2, self.ccpert_A.build_Aovoo_dp())
        self.polar2 -= ndot('bk,kb->', tmp, self.x1_B, prefactor=0.5)
        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = ndot('ijab,kjab->ik', self.l2, self.x2_B)
        self.polar2 -= ndot('ik,ki->', tmp, self.ccpert_A.build_Aoo_dp(), prefactor=0.5)
        tmp = ndot('ijab,kiba->jk', self.l2, self.x2_B, )
        self.polar2 -= ndot('jk,kj->', tmp, self.ccpert_A.build_Aoo_dp(), prefactor=0.5)
        tmp = ndot('ijab,ijac->bc', self.l2, self.x2_B, )
        self.polar2 += ndot('bc,bc->', tmp, self.ccpert_A.build_Avv_dp(), prefactor=0.5)
        tmp = ndot('ijab,ijcb->ac', self.l2, self.x2_B, )
        self.polar2 += ndot('ac,ac->', tmp, self.ccpert_A.build_Avv_dp(), prefactor=0.5)

        return -1.0 * (self.polar1 + self.polar2)

# End HelperCCLinresp class
