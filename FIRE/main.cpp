#include <algorithm>
#include <chrono>
#include <iterator>
#include <functional>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <x86intrin.h>
//---------------------------------------------------------------
constexpr auto deg = 3;
constexpr auto N   = 1000;
constexpr auto rho = 1.204;

constexpr auto dt     = 5e-3;
constexpr auto margin = 0.5;

constexpr auto Ndof  = deg*N;
constexpr auto Ninv  = 1.0/N;
constexpr auto SKIN2 = (margin*0.5) * (margin*0.5);
constexpr auto dt2   = dt*0.5;
constexpr auto dt4   = dt*0.25;
constexpr auto N_A   = N*4/5;

const double Lbox = std::pow(N/rho, 1.0/deg);
const double Linv = 1.0/Lbox;

double __attribute__((aligned(32))) conf[N][4], velo[N][4], force[N][4], NL_config[N][4];
int point[N], list[N*100];
double vxi1 = 0.0;

const __m256d uhalf = _mm256_set1_pd(0.5);
const __m256d vzero = _mm256_set1_pd(0.0);
const __m256d vone  = _mm256_set1_pd(1.0);
const __m256d vonha = _mm256_set1_pd(1.5);
const __m256d vtwo  = _mm256_set1_pd(2.0);
const __m256d v12   = _mm256_set1_pd(12.0);
const __m256d vLinv = _mm256_set1_pd(Linv);
const __m256d vLbox = _mm256_set1_pd(Lbox);
const __m256d vmarg = _mm256_set1_pd(margin);
const __m256d vrc2diff1 = _mm256_set1_pd(2.25);// = 1.5**2
const __m256d vrc2diff2 = _mm256_set1_pd(1.75);
const __m256d vs6diff1  = _mm256_set1_pd(0.535595913216);// = 1.0 - 0.88**6
const __m256d vs6diff2  = _mm256_set1_pd(0.737856);// = 1.0 - 0.8**6

enum {X, Y, Z};
//---------------------------------------------------------------
void init_lattice() {
    const auto ln   = std::ceil(std::pow(N, 1.0/deg));
    const auto haba = Lbox/ln;
    const auto lnz  = std::ceil(N/(ln*ln));
    const auto zaba = Lbox/lnz;

    for (int i=0; i<N; i++) {
        int iz = std::floor(i/(ln*ln));
        int iy = std::floor((i - iz*ln*ln)/ln);
        int ix = i - iz*ln*ln - iy*ln;

        conf[i][X] = haba*0.5 + haba * ix;
        conf[i][Y] = haba*0.5 + haba * iy;
        conf[i][Z] = zaba*0.5 + zaba * iz;

        for (int d=0; d<deg; d++) {
            conf[i][d] -= Lbox * std::round(conf[i][d] * Linv);
        }
    }
}
inline void remove_drift() {
    double vel1 = 0.0, vel2 = 0.0, vel3 = 0.0;
    for (int i=0; i<N; i++) {
        vel1 += velo[i][X];
        vel2 += velo[i][Y];
        vel3 += velo[i][Z];
    }
    vel1 *= Ninv;
    vel2 *= Ninv;
    vel3 *= Ninv;
    for (int i=0; i<N; i++) {
        velo[i][X] -= vel1;
        velo[i][Y] -= vel2;
        velo[i][Z] -= vel3;
    }
}
void init_vel_MB(const double T_targ, std::mt19937 &mt) {
    std::normal_distribution<double> dist_trans(0.0, std::sqrt(T_targ));
    for (int i=0; i<N; i++) {
        velo[i][X] = dist_trans(mt);
        velo[i][Y] = dist_trans(mt);
        velo[i][Z] = dist_trans(mt);
    }
    remove_drift();
}
void init_species(std::mt19937 &mt) {
    std::vector<int> v(N);
    std::iota(v.begin(), v.end(), 0);
    std::shuffle(v.begin(), v.end(), mt);
    for (int i=0; i<N; i+=2) {
        const int id0 = v[i];
        const int id1 = v[i+1];
        const __m256d vi0 = _mm256_load_pd((double *)(conf + id0));
        const __m256d vi1 = _mm256_load_pd((double *)(conf + id1));
        _mm256_store_pd((double *)(conf + id1), vi0);
        _mm256_store_pd((double *)(conf + id0), vi1);
    }
}
//---------------------------------------------------------------
inline double KABLJ_energy(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 1.5;
        }
    case 1:
        switch (kj) {
        case 0:
            return 1.5;
        case 1:
            return 0.5;
        }
    }
    return 0.0;
}
inline double KABLJ_energy24(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 24.0;
        case 1:
            return 36.0;
        }
    case 1:
        switch (kj) {
        case 0:
            return 36.0;
        case 1:
            return 12.0;
        }
    }
    return 0.0;
}
inline double KABLJ_sij6(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.0;
        case 1:
            return 0.262144;// = 0.64 * 0.64 * 0.64
        }
    case 1:
        switch (kj) {
        case 0:
            return 0.2621440;
        case 1:
            return 0.464404086784;// = 0.7744 * 0.7744 * 0.7744
        }
    }
    return 0.0;
}
inline double MBLJ_rcut1(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 1.5;// = 1.5*sAA
        case 1:
            return 2.0;// = 2.5*sAB
        }
    case 1:
        switch (kj) {
        case 0:
            return 2.0;// = 2.5*sAB
        case 1:
            return 1.5;// = 1.5*sAA
        }
    }
    return 0.0;
}
inline double MBLJ_Aij(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return 4.8251201293589813e-02;// = -(2./(1.5**12) - 1./(1.5**6))/1.5
        case 1:
            return 2.0312227840000000e-03;// = -(2*(0.4**12) - 0.4**6)/2
        }
    case 1:
        switch (kj) {
        case 0:
            return 2.0312227840000000e-03;// = -(2*(0.4**12) - 0.4**6)/2
        case 1:
            return 2.4964149629031374e-02;// = -(2.0*(0.88/1.5)**12 - (0.88/1.5)**6)/1.5
        }
    }
    return 0.0;
}
inline double MBLJ_Bij(const int &ki, const int &kj) {
    switch (ki) {
    case 0:
        switch (kj) {
        case 0:
            return -3.2033659427857464e-01;// = 4.0*(1.0/1.5**12 - 1.0/1.5**6)
        case 1:
            return -1.6316891136000000e-02;// = 4.0*((0.8/2.0)**12 - (0.8/2.0)**6)
        }
    case 1:
        switch (kj) {
        case 0:
            return -1.6316891136000000e-02;// = 4.0*((0.8/2.0)**12 - (0.8/2.0)**6)
        case 1:
            return -1.5643390719759068e-01;// = 4.0*((0.88/1.5)**12 - (0.88/1.5)**6)
        }
    }
    return 0.0;
}
//---------------------------------------------------------------
inline double hsum_double_avx(__m256d v) {
    __m128d vlow  = _mm256_castpd256_pd128(v);
    __m128d vhigh = _mm256_extractf128_pd(v, 1);
    vlow = _mm_add_pd(vlow, vhigh);
    __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
    return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));
}
//---------------------------------------------------------------
constexpr auto Nend = N-1;
void generate_NL() {
    auto nlist = -1;
    for (int i=0; i<N-4; i++) {
        point[i] = nlist+1;

        const int ki = i>=N_A;
        const __m256d vspeci = _mm256_set1_pd(ki);
        const __m256d vi = _mm256_load_pd((double *)(conf + i));
        const int jstart = i+1;
        const int jend_tmp = jstart + (N - jstart)/4*4;

        // initial
        int ja0 = jstart;
        int ja1 = jstart+1;
        int ja2 = jstart+2;
        int ja3 = jstart+3;
        const __m256d vja0 = _mm256_load_pd((double *)(conf + ja0));
        const __m256d vja1 = _mm256_load_pd((double *)(conf + ja1));
        const __m256d vja2 = _mm256_load_pd((double *)(conf + ja2));
        const __m256d vja3 = _mm256_load_pd((double *)(conf + ja3));
        __m256d vija0 = vi - vja0;
        __m256d vija1 = vi - vja1;
        __m256d vija2 = vi - vja2;
        __m256d vija3 = vi - vja3;
        vija0 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vija0, vLinv, uhalf));
        vija1 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vija1, vLinv, uhalf));
        vija2 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vija2, vLinv, uhalf));
        vija3 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vija3, vLinv, uhalf));

        __m256d tmp0 = _mm256_unpacklo_pd(vija0, vija1);
        __m256d tmp1 = _mm256_unpackhi_pd(vija0, vija1);
        __m256d tmp2 = _mm256_unpacklo_pd(vija2, vija3);
        __m256d tmp3 = _mm256_unpackhi_pd(vija2, vija3);
        __m256d vdx = _mm256_permute2f128_pd(tmp0, tmp2, 2*16+1*0);
        __m256d vdy = _mm256_permute2f128_pd(tmp1, tmp3, 2*16+1*0);
        __m256d vdz = _mm256_permute2f128_pd(tmp0, tmp2, 3*16+1*1);
        __m256d vr2 = _mm256_fmadd_pd(vdx, vdx, _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
        double rij2a0 = vr2[0];
        double rij2a1 = vr2[1];
        double rij2a2 = vr2[2];
        double rij2a3 = vr2[3];

        int kja0 = ja0>=N_A;
        int kja1 = ja1>=N_A;
        int kja2 = ja2>=N_A;
        int kja3 = ja3>=N_A;
        __m256d vspecj = _mm256_set_pd(kja3, kja2, kja1, kja0);
        __m256d sdiff = vspecj - vspeci;
        __m256d vrc1 = _mm256_fmadd_pd(sdiff, uhalf, vonha);
        vrc1 = vrc1 + vmarg;
        vrc1 = vrc1 * vrc1;
        double rlist2a0 = vrc1[0];
        double rlist2a1 = vrc1[1];
        double rlist2a2 = vrc1[2];
        double rlist2a3 = vrc1[3];

        for (int j=jstart+4; j<jend_tmp; j+=4) {
            int jb0 = j;
            int jb1 = j+1;
            int jb2 = j+2;
            int jb3 = j+3;
            const __m256d vjb0 = _mm256_load_pd((double *)(conf + jb0));
            const __m256d vjb1 = _mm256_load_pd((double *)(conf + jb1));
            const __m256d vjb2 = _mm256_load_pd((double *)(conf + jb2));
            const __m256d vjb3 = _mm256_load_pd((double *)(conf + jb3));
            __m256d vijb0 = vi - vjb0;
            __m256d vijb1 = vi - vjb1;
            __m256d vijb2 = vi - vjb2;
            __m256d vijb3 = vi - vjb3;
            vijb0 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vijb0, vLinv, uhalf));
            vijb1 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vijb1, vLinv, uhalf));
            vijb2 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vijb2, vLinv, uhalf));
            vijb3 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vijb3, vLinv, uhalf));

            tmp0 = _mm256_unpacklo_pd(vijb0, vijb1);
            tmp1 = _mm256_unpackhi_pd(vijb0, vijb1);
            tmp2 = _mm256_unpacklo_pd(vijb2, vijb3);
            tmp3 = _mm256_unpackhi_pd(vijb2, vijb3);
            vdx = _mm256_permute2f128_pd(tmp0, tmp2, 2*16+1*0);
            vdy = _mm256_permute2f128_pd(tmp1, tmp3, 2*16+1*0);
            vdz = _mm256_permute2f128_pd(tmp0, tmp2, 3*16+1*1);
            vr2 = _mm256_fmadd_pd(vdx, vdx, _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));

            int kjb0 = jb0>=N_A;
            int kjb1 = jb1>=N_A;
            int kjb2 = jb2>=N_A;
            int kjb3 = jb3>=N_A;
            __m256d vspecj = _mm256_set_pd(kjb3, kjb2, kjb1, kjb0);
            __m256d sdiff = vspecj - vspeci;
            __m256d vrc1 = _mm256_fmadd_pd(sdiff, uhalf, vonha);
            vrc1 = vrc1 + vmarg;
            vrc1 = vrc1 * vrc1;

            //----------------------------------------

            if (rij2a0 < rlist2a0) {
                nlist++;
                list[nlist] = ja0;
            }
            if (rij2a1 < rlist2a1) {
                nlist++;
                list[nlist] = ja1;
            }
            if (rij2a2 < rlist2a2) {
                nlist++;
                list[nlist] = ja2;
            }
            if (rij2a3 < rlist2a3) {
                nlist++;
                list[nlist] = ja3;
            }

            //----------------------------------------

            ja0 = jb0;
            ja1 = jb1;
            ja2 = jb2;
            ja3 = jb3;
            rij2a0 = vr2[0];
            rij2a1 = vr2[1];
            rij2a2 = vr2[2];
            rij2a3 = vr2[3];
            rlist2a0 = vrc1[0];
            rlist2a1 = vrc1[1];
            rlist2a2 = vrc1[2];
            rlist2a3 = vrc1[3];
        }

        // final
        if (rij2a0 < rlist2a0) {
            nlist++;
            list[nlist] = ja0;
        }
        if (rij2a1 < rlist2a1) {
            nlist++;
            list[nlist] = ja1;
        }
        if (rij2a2 < rlist2a2) {
            nlist++;
            list[nlist] = ja2;
        }
        if (rij2a3 < rlist2a3) {
            nlist++;
            list[nlist] = ja3;
        }

        // gomi
        for (int j=jend_tmp; j<N; j++) {
            const int kj = j>=N_A;
            const __m256d vj = _mm256_load_pd((double *)(conf + j));
            __m256d vij = vi - vj;
            vij -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vij, vLinv, uhalf));
            double rij2 = hsum_double_avx(vij*vij);
            double rc1 = MBLJ_rcut1(ki, kj) + margin;
            double rlist2 = rc1 * rc1;
            if (rij2 < rlist2) {
                nlist++;
                list[nlist] = j;
            }
        }
    }
    for (int i=N-4; i<N; i++) {
        point[i] = nlist+1;
        const int ki = i>=N_A;
        const __m256d vi = _mm256_load_pd((double *)(conf + i));
        for (int j=i+1; j<N; j++) {
            const int kj = j>=N_A;
            const __m256d vj = _mm256_load_pd((double *)(conf + j));
            __m256d vij = vi - vj;
            vij -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vij, vLinv, uhalf));
            double rij2 = hsum_double_avx(vij*vij);
            double rc1 = MBLJ_rcut1(ki, kj) + margin;
            double rlist2 = rc1 * rc1;
            if (rij2 < rlist2) {
                nlist++;
                list[nlist] = j;
            }
        }
    }
    std::copy(*conf, *conf+Ndof+N, *NL_config);
}
//---------------------------------------------------------------
void calc_force() {
    std::fill(*force, *force+Ndof+N, 0.0);
    for (int i=0; i<Nend; i++) {
        const int pstart = point[i];
        const int pend = point[i+1];
        if (pstart == pend) continue;
        const int pend_tmp = pstart+(pend-pstart)/4*4;

        const int si = i>=N_A;
        const __m256d vspeci = _mm256_set1_pd(si);
        const __m256d vqi = _mm256_load_pd((double *)(conf + i));
        __m256d vfi = _mm256_load_pd((double *)(force + i));

        // initial
        int ja0 = list[pstart];
        int ja1 = list[pstart+1];
        int ja2 = list[pstart+2];
        int ja3 = list[pstart+3];

        __m256d vqja0 = _mm256_load_pd((double *)(conf + ja0));
        __m256d vqja1 = _mm256_load_pd((double *)(conf + ja1));
        __m256d vqja2 = _mm256_load_pd((double *)(conf + ja2));
        __m256d vqja3 = _mm256_load_pd((double *)(conf + ja3));
        __m256d vdra0 = vqi - vqja0;
        __m256d vdra1 = vqi - vqja1;
        __m256d vdra2 = vqi - vqja2;
        __m256d vdra3 = vqi - vqja3;
        vdra0 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdra0, vLinv, uhalf));
        vdra1 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdra1, vLinv, uhalf));
        vdra2 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdra2, vLinv, uhalf));
        vdra3 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdra3, vLinv, uhalf));

        __m256d tmp0 = _mm256_unpacklo_pd(vdra0, vdra1);
        __m256d tmp1 = _mm256_unpackhi_pd(vdra0, vdra1);
        __m256d tmp2 = _mm256_unpacklo_pd(vdra2, vdra3);
        __m256d tmp3 = _mm256_unpackhi_pd(vdra2, vdra3);
        __m256d vdx = _mm256_permute2f128_pd(tmp0, tmp2, 2*16+1*0);
        __m256d vdy = _mm256_permute2f128_pd(tmp1, tmp3, 2*16+1*0);
        __m256d vdz = _mm256_permute2f128_pd(tmp0, tmp2, 3*16+1*1);
        __m256d vr2 = _mm256_fmadd_pd(vdx, vdx, _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));

        int sj0 = ja0>=N_A;
        int sj1 = ja1>=N_A;
        int sj2 = ja2>=N_A;
        int sj3 = ja3>=N_A;
        __m256d vspecj = _mm256_set_pd(sj3, sj2, sj1, sj0);
        __m256d sdiff = vspecj - vspeci;
        __m256d sadd = _mm256_floor_pd(uhalf*(vspeci + vspecj));

        __m256d vs6 = -_mm256_fmadd_pd(vs6diff1, sadd,
                                       _mm256_fmsub_pd(vs6diff2, sdiff, vone));
        __m256d eij = v12 * (vtwo + sdiff - sadd);

        double Aij_0 = MBLJ_Aij(si, sj0);
        double Aij_1 = MBLJ_Aij(si, sj1);
        double Aij_2 = MBLJ_Aij(si, sj2);
        double Aij_3 = MBLJ_Aij(si, sj3);
        __m256d Aij = _mm256_set_pd(Aij_3, Aij_2, Aij_1, Aij_0);

        __m256d vr6 = vr2 * vr2 * vr2;
        __m256d vr14 = vr6 * vr6 * vr2;
        __m256d vr1 = _mm256_sqrt_pd(vr2);
        __m256d df = -eij * (vr1 * vs6 * _mm256_fmsub_pd(vtwo, vs6, vr6)
                             + vr14 * Aij)/(vr14 * vr1);

        __m256d rcut2 = _mm256_fmadd_pd(sdiff, vrc2diff2, vrc2diff1);
        __m256d mask = rcut2 - vr2;
        df = _mm256_blendv_pd(df, vzero, mask);
        if (pend-pstart < 4) df = _mm256_setzero_pd();

        for (int p=pstart+4; p<pend_tmp; p+=4) {
            int jb0 = list[p];
            int jb1 = list[p+1];
            int jb2 = list[p+2];
            int jb3 = list[p+3];
            __m256d vqjb0 = _mm256_load_pd((double *)(conf + jb0));
            __m256d vqjb1 = _mm256_load_pd((double *)(conf + jb1));
            __m256d vqjb2 = _mm256_load_pd((double *)(conf + jb2));
            __m256d vqjb3 = _mm256_load_pd((double *)(conf + jb3));
            __m256d vdrb0 = vqi - vqjb0;
            __m256d vdrb1 = vqi - vqjb1;
            __m256d vdrb2 = vqi - vqjb2;
            __m256d vdrb3 = vqi - vqjb3;
            vdrb0 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdrb0, vLinv, uhalf));
            vdrb1 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdrb1, vLinv, uhalf));
            vdrb2 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdrb2, vLinv, uhalf));
            vdrb3 -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vdrb3, vLinv, uhalf));

            tmp0 = _mm256_unpacklo_pd(vdrb0, vdrb1);
            tmp1 = _mm256_unpackhi_pd(vdrb0, vdrb1);
            tmp2 = _mm256_unpacklo_pd(vdrb2, vdrb3);
            tmp3 = _mm256_unpackhi_pd(vdrb2, vdrb3);
            vdx = _mm256_permute2f128_pd(tmp0, tmp2, 2*16+1*0);
            vdy = _mm256_permute2f128_pd(tmp1, tmp3, 2*16+1*0);
            vdz = _mm256_permute2f128_pd(tmp0, tmp2, 3*16+1*1);
            vr2 = _mm256_fmadd_pd(vdx, vdx, _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));

            sj0 = jb0>=N_A;
            sj1 = jb1>=N_A;
            sj2 = jb2>=N_A;
            sj3 = jb3>=N_A;
            vspecj = _mm256_set_pd(sj3, sj2, sj1, sj0);
            sdiff = vspecj - vspeci;
            sadd = _mm256_floor_pd(uhalf*(vspeci + vspecj));

            vr6 = vr2 * vr2 * vr2;
            vr14 = vr6 * vr6 * vr2;
            vr1 = _mm256_sqrt_pd(vr2);

            rcut2 = _mm256_fmadd_pd(sdiff, vrc2diff2, vrc2diff1);
            mask = rcut2 - vr2;

            vs6 = -_mm256_fmadd_pd(vs6diff1, sadd,
                                   _mm256_fmsub_pd(vs6diff2, sdiff, vone));
            eij = v12 * (vtwo + sdiff - sadd);

            Aij_0 = MBLJ_Aij(si, sj0);
            Aij_1 = MBLJ_Aij(si, sj1);
            Aij_2 = MBLJ_Aij(si, sj2);
            Aij_3 = MBLJ_Aij(si, sj3);
            Aij = _mm256_set_pd(Aij_3, Aij_2, Aij_1, Aij_0);

            //----------------------------------------

            __m256d vdf_0 = _mm256_permute4x64_pd(df, 0);
            __m256d vdf_1 = _mm256_permute4x64_pd(df, 85);
            __m256d vdf_2 = _mm256_permute4x64_pd(df, 170);
            __m256d vdf_3 = _mm256_permute4x64_pd(df, 255);
            __m256d vpj_0 = _mm256_load_pd((double *)(force + ja0));
            __m256d vpj_1 = _mm256_load_pd((double *)(force + ja1));
            __m256d vpj_2 = _mm256_load_pd((double *)(force + ja2));
            __m256d vpj_3 = _mm256_load_pd((double *)(force + ja3));

            vfi = -_mm256_fmsub_pd(vdf_0, vdra0, vfi);
            vfi = -_mm256_fmsub_pd(vdf_1, vdra1, vfi);
            vfi = -_mm256_fmsub_pd(vdf_2, vdra2, vfi);
            vfi = -_mm256_fmsub_pd(vdf_3, vdra3, vfi);
            vpj_0 = _mm256_fmadd_pd(vdf_0, vdra0, vpj_0);
            vpj_1 = _mm256_fmadd_pd(vdf_1, vdra1, vpj_1);
            vpj_2 = _mm256_fmadd_pd(vdf_2, vdra2, vpj_2);
            vpj_3 = _mm256_fmadd_pd(vdf_3, vdra3, vpj_3);

            _mm256_store_pd((double *)(force + ja0), vpj_0);
            _mm256_store_pd((double *)(force + ja1), vpj_1);
            _mm256_store_pd((double *)(force + ja2), vpj_2);
            _mm256_store_pd((double *)(force + ja3), vpj_3);

            //----------------------------------------

            ja0 = jb0;
            ja1 = jb1;
            ja2 = jb2;
            ja3 = jb3;
            vdra0 = vdrb0;
            vdra1 = vdrb1;
            vdra2 = vdrb2;
            vdra3 = vdrb3;
            df = -eij * (vr1 * vs6 * _mm256_fmsub_pd(vtwo, vs6, vr6)
                         + vr14 * Aij)/(vr14 * vr1);
            df = _mm256_blendv_pd(df, vzero, mask);
        }

        // final
        __m256d vdf_0 = _mm256_permute4x64_pd(df, 0);
        __m256d vdf_1 = _mm256_permute4x64_pd(df, 85);
        __m256d vdf_2 = _mm256_permute4x64_pd(df, 170);
        __m256d vdf_3 = _mm256_permute4x64_pd(df, 255);
        __m256d vpj_0 = _mm256_load_pd((double *)(force + ja0));
        __m256d vpj_1 = _mm256_load_pd((double *)(force + ja1));
        __m256d vpj_2 = _mm256_load_pd((double *)(force + ja2));
        __m256d vpj_3 = _mm256_load_pd((double *)(force + ja3));

        vfi = -_mm256_fmsub_pd(vdf_0, vdra0, vfi);
        vfi = -_mm256_fmsub_pd(vdf_1, vdra1, vfi);
        vfi = -_mm256_fmsub_pd(vdf_2, vdra2, vfi);
        vfi = -_mm256_fmsub_pd(vdf_3, vdra3, vfi);
        vpj_0 = _mm256_fmadd_pd(vdf_0, vdra0, vpj_0);
        vpj_1 = _mm256_fmadd_pd(vdf_1, vdra1, vpj_1);
        vpj_2 = _mm256_fmadd_pd(vdf_2, vdra2, vpj_2);
        vpj_3 = _mm256_fmadd_pd(vdf_3, vdra3, vpj_3);

        _mm256_store_pd((double *)(force + ja0), vpj_0);
        _mm256_store_pd((double *)(force + ja1), vpj_1);
        _mm256_store_pd((double *)(force + ja2), vpj_2);
        _mm256_store_pd((double *)(force + ja3), vpj_3);

        //------------------------------

        // gomi
        for (int p=pend_tmp; p<pend; p++) {
            const int j = list[p];
            const int sj = j>=N_A;
            const __m256d vj = _mm256_load_pd((double *)(conf + j));
            __m256d vij = vqi - vj;
            vij -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vij, vLinv, uhalf));

            const double rij2 = hsum_double_avx(vij*vij);
            const double rc2  = 2.25 + 1.75*(sj - si);
            if (rij2 < rc2) {
                double sij6 = KABLJ_sij6(si, sj);
                double rij1 = sqrt(rij2);
                double rij6 = rij2 * rij2 * rij2;
                double rij14 = rij6 * rij6 * rij2;
                double cAij = MBLJ_Aij(si, sj);

                double temp = -(rij1 * sij6 * (2.0*sij6 - rij6) + cAij * rij14)/(rij14 * rij1);
                temp *= KABLJ_energy24(si, sj);

                __m256d vtemp = _mm256_set1_pd(temp);
                vfi -= vtemp * vij;
                __m256d vfj = _mm256_load_pd((double *)(force + j));
                vfj += vtemp * vij;
                _mm256_store_pd((double *)(force + j), vfj);
            }
        }
        _mm256_store_pd((double *)(force + i), vfi);
    }
}
//---------------------------------------------------------------
double calc_potential() {
    double ans = 0.0;
    for (int i=0; i<Nend; i++) {
        const int si = i>=N_A;
        const int pend = point[i+1];
        const double xi = conf[i][X];
        const double yi = conf[i][Y];
        const double zi = conf[i][Z];
        for (int p=point[i]; p<pend; p++) {
            const int j = list[p];
            const int sj = j>=N_A;

            double dx = xi - conf[j][X];
            double dy = yi - conf[j][Y];
            double dz = zi - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = dx*dx + dy*dy + dz*dz;
            double rc2  = 2.25 + 1.75*abs(si - sj);
            if (rij2 < rc2) {
                double sij6 = KABLJ_sij6(si, sj);
                double rij6 = rij2 * rij2 * rij2;
                double Aij = 24.0 * MBLJ_Aij(si, sj);
                double Bij = MBLJ_Bij(si, sj);
                double temp = 4.0 * sij6 * (sij6 - rij6)/(rij6 * rij6) - Bij - Aij * (sqrt(rij2) - sqrt(rc2));
                ans += KABLJ_energy(si, sj)*temp;
            }
        }
    }
    return ans;
}
double calc_potential_N2() {
    double ans = 0.0;
    for (int i=0; i<N; i++) {
        const int si = i>=N_A;
        const double xi = conf[i][X];
        const double yi = conf[i][Y];
        const double zi = conf[i][Z];
        for (int j=i+1; j<N; j++) {
            const int sj = j>=N_A;

            double dx = xi - conf[j][X];
            double dy = yi - conf[j][Y];
            double dz = zi - conf[j][Z];
            dx -= Lbox * floor(dx * Linv + 0.5);
            dy -= Lbox * floor(dy * Linv + 0.5);
            dz -= Lbox * floor(dz * Linv + 0.5);

            double rij2 = dx*dx + dy*dy + dz*dz;
            double rc2  = 2.25 + 1.75*abs(si - sj);
            if (rij2 < rc2) {
                double sij6 = KABLJ_sij6(si, sj);
                double rij6 = rij2 * rij2 * rij2;
                double Aij = 24.0 * MBLJ_Aij(si, sj);
                double Bij = MBLJ_Bij(si, sj);
                double temp = 4.0 * sij6 * (sij6 - rij6)/(rij6 * rij6) - Bij - Aij * (sqrt(rij2) - sqrt(rc2));
                ans += KABLJ_energy(si, sj)*temp;
            }
        }
    }
    return ans;
}
//---------------------------------------------------------------
const __m256d vdt1 = _mm256_set1_pd(dt);
const __m256d vdt2 = _mm256_set1_pd(dt2);
inline void velocity_update() {
    for (int i=0; i<N; i++) {
        __m256d vi = _mm256_load_pd((double *)(velo + i));
        __m256d fi = _mm256_load_pd((double *)(force + i));
        vi = _mm256_fmadd_pd(fi, vdt2, vi);
        _mm256_store_pd((double *)(velo + i), vi);
    }
}
inline void position_update() {
    for (int i=0; i<N; i++) {
        __m256d vi = _mm256_load_pd((double *)(velo + i));
        __m256d ri = _mm256_load_pd((double *)(conf + i));
        ri = _mm256_fmadd_pd(vi, vdt1, ri);
        _mm256_store_pd((double *)(conf + i), ri);
    }
}
inline void PBC() {
    for (int i=0; i<N; i++) {
        __m256d ri = _mm256_load_pd((double *)(conf + i));
        ri -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(ri, vLinv, uhalf));
        _mm256_store_pd((double *)(conf + i), ri);
    }
}
inline void NL_check() {
    double dev_max = 0.0;
    for (int i=0; i<N; i++) {
        __m256d rit = _mm256_load_pd((double *)(conf + i));
        __m256d ri0 = _mm256_load_pd((double *)(NL_config + i));
        __m256d vij = rit - ri0;
        vij -= vLbox * _mm256_floor_pd(_mm256_fmadd_pd(vij, vLinv, uhalf));
        double rij2 = hsum_double_avx(vij*vij);
        if (rij2 > dev_max) dev_max = rij2;
    }
    if (dev_max > SKIN2) {// renew neighbor list
        generate_NL();
    }
}
//---------------------------------------------------------------
void NVT(const double T_targ, const double tsim) {
    calc_force();
    // Nose-Hoover variables
    const auto gkBT = Ndof*T_targ;

    long t = 0;
    const long steps = tsim/dt;
    while (t < steps) {
        // Nose-Hoover chain (QMASS = 1.0)
        double uk = std::inner_product(*velo, *velo+Ndof+N, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        double temp = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof+N, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        velocity_update();
        position_update();
        PBC();
        NL_check();
        calc_force();
        velocity_update();

        // Nose-Hoover chain (QMASS = 1.0)
        uk    = std::inner_product(*velo, *velo+Ndof+N, *velo, 0.0);
        vxi1 += dt4 * (uk - gkBT);
        temp  = std::exp(-vxi1 * dt2);
        std::transform(*velo, *velo+Ndof+N, *velo,
                       std::bind(std::multiplies<double>(), std::placeholders::_1, temp));
        vxi1 += dt4 * (uk*temp*temp - gkBT);

        t++;
        if (!(t & 127)) remove_drift();
    }
}
//---------------------------------------------------------------
inline double get_fmax() {
    double fmax = 0.0;
    for (int i=0; i<N; i++) {
        double fi = std::inner_product(*force+4*i, *force+4*i+deg, *force+4*i, 0.0);
        if (fmax < fi) fmax = fi;
    }
    return sqrt(fmax);
}
void FIRE() {
    calc_force();

    ////// FIRE variables //////
    constexpr auto alpha_fire_0 = 0.1;
    constexpr auto finc         = 1.1;
    constexpr auto fdec         = 0.5;
    constexpr auto falpha       = 0.99;
    constexpr auto Nmin         = 5;

    auto Nppos = 0;
    auto dt    = 4e-3;
    auto alpha_fire = alpha_fire_0;
    const auto dtmax = dt*5;

    constexpr auto fmax_tol = 1e-12;
    //////////////////////////////

    std::fill(*velo, *velo+Ndof+N, 0.0);

    std::ofstream out("log.dat");
    double fmax = get_fmax();
    out << std::setprecision(6) << std::scientific
        << fmax << "," << Nppos << std::endl;

    auto t = 0;
    while (1) {
        /////// FIRE ///////
        double P = std::inner_product(*velo, *velo+Ndof+N, *force, 0.0);
        if (P > 0.0) {
            Nppos += 1;
            // mixing
            double Fnorm = std::inner_product(*velo, *velo+Ndof+N, *velo, 0.0) / std::inner_product(*force, *force+Ndof+N, *force, 0.0);
            Fnorm = std::sqrt(Fnorm);
            for (int i=0; i<N; i++) {
                velo[i][X] = (1.0 - alpha_fire) * velo[i][X] + alpha_fire*Fnorm * force[i][X];
                velo[i][Y] = (1.0 - alpha_fire) * velo[i][Y] + alpha_fire*Fnorm * force[i][Y];
                velo[i][Z] = (1.0 - alpha_fire) * velo[i][Z] + alpha_fire*Fnorm * force[i][Z];
            }

            if (Nppos > Nmin) {
                dt = std::min(dt*finc, dtmax);
                alpha_fire *= falpha;
            }
        } else if (P <= 0.0) {
            Nppos = 0;
            std::fill(*velo, *velo+Ndof+N, 0.0);
            dt *= fdec;
            alpha_fire = alpha_fire_0;
        }
        ////// FIRE end //////

        ////// MD //////
        // velocity update
        for (int i=0; i<N; i++) {
            velo[i][X] += dt*0.5*force[i][X];
            velo[i][Y] += dt*0.5*force[i][Y];
            velo[i][Z] += dt*0.5*force[i][Z];
        }

        // position update
        for (int i=0; i<N; i++) {
            conf[i][X] += dt*velo[i][X];
            conf[i][Y] += dt*velo[i][Y];
            conf[i][Z] += dt*velo[i][Z];
        }

        PBC();
        NL_check();
        calc_force();

        // velocity update
        for (int i=0; i<N; i++) {
            velo[i][X] += dt*0.5*force[i][X];
            velo[i][Y] += dt*0.5*force[i][Y];
            velo[i][Z] += dt*0.5*force[i][Z];
        }
        ////// MD end //////

        ////// converge //////
        double fmax = get_fmax();
        out << std::setprecision(6) << std::scientific
            << fmax << "," << Nppos << std::endl;
        if (fmax < fmax_tol) {
            std::cout << "======================\nFIRE end\n"
                      << t << " steps\nfmax = "
                      << std::setprecision(8) << std::scientific
                      << fmax << std::endl;
            return;
        }
        ////// converge end //////
        t++;
    }
}
int main(int argc, char** argv) {
    // initialize system
    std::fill(*conf, *conf+Ndof+N, 0.0);
    std::fill(*velo, *velo+Ndof+N, 0.0);

    std::mt19937 mt(20230126);
    init_lattice();
    init_species(mt);
    init_vel_MB(1.0, mt);

    // construct neighbor list
    generate_NL();

    // equilibration
    NVT(1.00, 2e2);
    NVT(0.50, 3e3);

    // FIRE
    FIRE();
}
