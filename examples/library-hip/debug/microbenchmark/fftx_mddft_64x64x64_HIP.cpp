
/*
 * This code was generated by Spiral 8.4.1.1-dev, www.spiral.net
 */

#include <stdint.h>
#include "hip/hip_runtime.h"
__device__ double  *P1, *P2;
 __device__ double D3[128] = {1.0, 0.0, 1.0, 0.0, 
      1.0, 0.0, 1.0, 0.0, 
      1.0, 0.0, 1.0, 0.0, 
      1.0, 0.0, 1.0, 0.0, 
      1.0, 0.0, 0.92387953251128674, (-0.38268343236508978), 
      0.99518472667219693, (-0.098017140329560604), 0.88192126434835505, (-0.47139673682599764), 
      0.98078528040323043, (-0.19509032201612825), 0.83146961230254524, (-0.55557023301960218), 
      0.95694033573220882, (-0.29028467725446233), 0.77301045336273699, (-0.63439328416364549), 
      1.0, 0.0, 0.70710678118654757, (-0.70710678118654757), 
      0.98078528040323043, (-0.19509032201612825), 0.55557023301960218, (-0.83146961230254524), 
      0.92387953251128674, (-0.38268343236508978), 0.38268343236508978, (-0.92387953251128674), 
      0.83146961230254524, (-0.55557023301960218), 0.19509032201612825, (-0.98078528040323043), 
      1.0, 0.0, 0.38268343236508978, (-0.92387953251128674), 
      0.95694033573220882, (-0.29028467725446233), 0.098017140329560604, (-0.99518472667219693), 
      0.83146961230254524, (-0.55557023301960218), (-0.19509032201612825), (-0.98078528040323043), 
      0.63439328416364549, (-0.77301045336273699), (-0.47139673682599764), (-0.88192126434835505), 
      1.0, 0.0, 0.0, (-1.0), 
      0.92387953251128674, (-0.38268343236508978), (-0.38268343236508978), (-0.92387953251128674), 
      0.70710678118654757, (-0.70710678118654757), (-0.70710678118654757), (-0.70710678118654757), 
      0.38268343236508978, (-0.92387953251128674), (-0.92387953251128674), (-0.38268343236508978), 
      1.0, 0.0, (-0.38268343236508978), (-0.92387953251128674), 
      0.88192126434835505, (-0.47139673682599764), (-0.77301045336273699), (-0.63439328416364549), 
      0.55557023301960218, (-0.83146961230254524), (-0.98078528040323043), (-0.19509032201612825), 
      0.098017140329560604, (-0.99518472667219693), (-0.95694033573220882), 0.29028467725446233, 
      1.0, 0.0, (-0.70710678118654757), (-0.70710678118654757), 
      0.83146961230254524, (-0.55557023301960218), (-0.98078528040323043), (-0.19509032201612825), 
      0.38268343236508978, (-0.92387953251128674), (-0.92387953251128674), 0.38268343236508978, 
      (-0.19509032201612825), (-0.98078528040323043), (-0.55557023301960218), 0.83146961230254524, 
      1.0, 0.0, (-0.92387953251128674), (-0.38268343236508978), 
      0.77301045336273699, (-0.63439328416364549), (-0.95694033573220882), 0.29028467725446233, 
      0.19509032201612825, (-0.98078528040323043), (-0.55557023301960218), 0.83146961230254524, 
      (-0.47139673682599764), (-0.88192126434835505), 0.098017140329560604, 0.99518472667219693};


#if 0
#define LAUNCHBOUNDS __launch_bounds__(128, 1)
#else
#define LAUNCHBOUNDS
#endif

#if 0
__global__ void LAUNCHBOUNDS ker_fftx_mddft_64x64x64_HIP0(double  *X) {
    __shared__ double T3[2048];
    double a238, a239, a240, a241, s61, s62, s63, s64, 
            s65, s66, s67, s68, s69, s70, s71, s72, 
            s73, s74, s75, s76, s77, s78, s79, s80, 
            t269, t270, t271, t272, t273, t274, t275, t276, 
            t277, t278, t279, t280, t281, t282, t283, t284, 
            t285, t286, t287, t288, t289, t290, t291, t292, 
            t293, t294, t295, t296;
    int a235, a236, a237, a242;
    a235 = (128*(threadIdx.x / 8));
    a236 = (threadIdx.x % 8);
    a237 = ((2048*blockIdx.x) + a235 + (2*a236));
    s61 = X[a237];
    s62 = X[(a237 + 1)];
    s63 = X[(a237 + 64)];
    s64 = X[(a237 + 65)];
    t269 = (s61 + s63);
    t270 = (s62 + s64);
    t271 = (s61 - s63);
    t272 = (s62 - s64);
    s65 = X[(a237 + 16)];
    s66 = X[(a237 + 17)];
    s67 = X[(a237 + 80)];
    s68 = X[(a237 + 81)];
    t273 = (s65 + s67);
    t274 = (s66 + s68);
    a238 = (0.70710678118654757*(s65 - s67));
    a239 = (0.70710678118654757*(s66 - s68));
    s69 = (a238 + a239);
    s70 = (a239 - a238);
    s71 = X[(a237 + 32)];
    s72 = X[(a237 + 33)];
    s73 = X[(a237 + 96)];
    s74 = X[(a237 + 97)];
    t275 = (s71 + s73);
    t276 = (s72 + s74);
    t277 = (s71 - s73);
    t278 = (s72 - s74);
    s75 = X[(a237 + 48)];
    s76 = X[(a237 + 49)];
    s77 = X[(a237 + 112)];
    s78 = X[(a237 + 113)];
    t279 = (s75 + s77);
    t280 = (s76 + s78);
    a240 = (0.70710678118654757*(s76 - s78));
    a241 = (0.70710678118654757*(s75 - s77));
    s79 = (a240 - a241);
    s80 = (a241 + a240);
    t281 = (t269 + t275);
    t282 = (t270 + t276);
    t283 = (t269 - t275);
    t284 = (t270 - t276);
    t285 = (t273 + t279);
    t286 = (t274 + t280);
    t287 = (t273 - t279);
    t288 = (t274 - t280);
    a242 = (a235 + (16*a236));
    T3[a242] = (t281 + t285);
    T3[(a242 + 1)] = (t282 + t286);
    T3[(a242 + 8)] = (t281 - t285);
    T3[(a242 + 9)] = (t282 - t286);
    T3[(a242 + 4)] = (t283 + t288);
    T3[(a242 + 5)] = (t284 - t287);
    T3[(a242 + 12)] = (t283 - t288);
    T3[(a242 + 13)] = (t284 + t287);
    t289 = (t271 + t278);
    t290 = (t272 - t277);
    t291 = (t271 - t278);
    t292 = (t272 + t277);
    t293 = (s69 + s79);
    t294 = (s70 - s80);
    t295 = (s69 - s79);
    t296 = (s70 + s80);
    T3[(a242 + 2)] = (t289 + t293);
    T3[(a242 + 3)] = (t290 + t294);
    T3[(a242 + 10)] = (t289 - t293);
    T3[(a242 + 11)] = (t290 - t294);
    T3[(a242 + 6)] = (t291 + t296);
    T3[(a242 + 7)] = (t292 - t295);
    T3[(a242 + 14)] = (t291 - t296);
    T3[(a242 + 15)] = (t292 + t295);
    __syncthreads();
    double a722, a723, a724, a725, a726, a727, a728, a729, 
            a730, a731, a732, a733, a734, a735, a736, a737, 
            a738, a739, a740, a741, s189, s190, s191, s192, 
            s193, s194, s195, s196, s197, s198, s199, s200, 
            s201, s202, s203, s204, s205, s206, s207, s208, 
            s209, s210, s211, s212, s213, s214, s215, s216, 
            s217, s218, s219, s220, s221, s222, s223, s224, 
            t422, t423, t424, t425, t426, t427, t428, t429, 
            t430, t431, t432, t433, t434, t435, t436, t437, 
            t438, t439, t440, t441, t442, t443, t444, t445, 
            t446, t447, t448, t449;
    int a718, a719, a720, a721, a742;
    a718 = (threadIdx.x / 8);
    a719 = (threadIdx.x % 8);
    a720 = ((128*a718) + (2*a719));
    s189 = T3[a720];
    s190 = T3[(a720 + 1)];
    s191 = T3[(a720 + 64)];
    s192 = T3[(a720 + 65)];
    a721 = (16*a719);
    a722 = D3[a721];
    a723 = D3[(a721 + 1)];
    s193 = ((a722*s189) - (a723*s190));
    s194 = ((a723*s189) + (a722*s190));
    a724 = D3[(a721 + 2)];
    a725 = D3[(a721 + 3)];
    s195 = ((a724*s191) - (a725*s192));
    s196 = ((a725*s191) + (a724*s192));
    t422 = (s193 + s195);
    t423 = (s194 + s196);
    t424 = (s193 - s195);
    t425 = (s194 - s196);
    s197 = T3[(a720 + 16)];
    s198 = T3[(a720 + 17)];
    s199 = T3[(a720 + 80)];
    s200 = T3[(a720 + 81)];
    a726 = D3[(a721 + 4)];
    a727 = D3[(5 + a721)];
    s201 = ((a726*s197) - (a727*s198));
    s202 = ((a727*s197) + (a726*s198));
    a728 = D3[(6 + a721)];
    a729 = D3[(7 + a721)];
    s203 = ((a728*s199) - (a729*s200));
    s204 = ((a729*s199) + (a728*s200));
    t426 = (s201 + s203);
    t427 = (s202 + s204);
    a730 = (0.70710678118654757*(s201 - s203));
    a731 = (0.70710678118654757*(s202 - s204));
    s205 = (a730 + a731);
    s206 = (a731 - a730);
    s207 = T3[(a720 + 32)];
    s208 = T3[(a720 + 33)];
    s209 = T3[(a720 + 96)];
    s210 = T3[(a720 + 97)];
    a732 = D3[(a721 + 8)];
    a733 = D3[(9 + a721)];
    s211 = ((a732*s207) - (a733*s208));
    s212 = ((a733*s207) + (a732*s208));
    a734 = D3[(10 + a721)];
    a735 = D3[(11 + a721)];
    s213 = ((a734*s209) - (a735*s210));
    s214 = ((a735*s209) + (a734*s210));
    t428 = (s211 + s213);
    t429 = (s212 + s214);
    t430 = (s211 - s213);
    t431 = (s212 - s214);
    s215 = T3[(a720 + 48)];
    s216 = T3[(a720 + 49)];
    s217 = T3[(a720 + 112)];
    s218 = T3[(a720 + 113)];
    a736 = D3[(a721 + 12)];
    a737 = D3[(13 + a721)];
    s219 = ((a736*s215) - (a737*s216));
    s220 = ((a737*s215) + (a736*s216));
    a738 = D3[(14 + a721)];
    a739 = D3[(15 + a721)];
    s221 = ((a738*s217) - (a739*s218));
    s222 = ((a739*s217) + (a738*s218));
    t432 = (s219 + s221);
    t433 = (s220 + s222);
    a740 = (0.70710678118654757*(s220 - s222));
    a741 = (0.70710678118654757*(s219 - s221));
    s223 = (a740 - a741);
    s224 = (a741 + a740);
    t434 = (t422 + t428);
    t435 = (t423 + t429);
    t436 = (t422 - t428);
    t437 = (t423 - t429);
    t438 = (t426 + t432);
    t439 = (t427 + t433);
    t440 = (t426 - t432);
    t441 = (t427 - t433);
    a742 = ((32*blockIdx.x) + (8192*a719) + (2*a718));
    P1[a742] = (t434 + t438);
    P1[(a742 + 1)] = (t435 + t439);
    P1[(a742 + 262144)] = (t434 - t438);
    P1[(a742 + 262145)] = (t435 - t439);
    P1[(a742 + 131072)] = (t436 + t441);
    P1[(a742 + 131073)] = (t437 - t440);
    P1[(a742 + 393216)] = (t436 - t441);
    P1[(a742 + 393217)] = (t437 + t440);
    t442 = (t424 + t431);
    t443 = (t425 - t430);
    t444 = (t424 - t431);
    t445 = (t425 + t430);
    t446 = (s205 + s223);
    t447 = (s206 - s224);
    t448 = (s205 - s223);
    t449 = (s206 + s224);
    P1[(a742 + 65536)] = (t442 + t446);
    P1[(a742 + 65537)] = (t443 + t447);
    P1[(a742 + 327680)] = (t442 - t446);
    P1[(a742 + 327681)] = (t443 - t447);
    P1[(a742 + 196608)] = (t444 + t449);
    P1[(a742 + 196609)] = (t445 - t448);
    P1[(a742 + 458752)] = (t444 - t449);
    P1[(a742 + 458753)] = (t445 + t448);
    __syncthreads();
}

__global__ void LAUNCHBOUNDS ker_fftx_mddft_64x64x64_HIP1() {
    __shared__ double T25[2048];
    double a980, a981, a982, a983, s285, s286, s287, s288, 
            s289, s290, s291, s292, s293, s294, s295, s296, 
            s297, s298, s299, s300, s301, s302, s303, s304, 
            t574, t575, t576, t577, t578, t579, t580, t581, 
            t582, t583, t584, t585, t586, t587, t588, t589, 
            t590, t591, t592, t593, t594, t595, t596, t597, 
            t598, t599, t600, t601;
    int a977, a978, a979, a984;
    a977 = (128*(threadIdx.x / 8));
    a978 = (threadIdx.x % 8);
    a979 = ((2048*blockIdx.x) + a977 + (2*a978));
    s285 = P1[a979];
    s286 = P1[(a979 + 1)];
    s287 = P1[(a979 + 64)];
    s288 = P1[(a979 + 65)];
    t574 = (s285 + s287);
    t575 = (s286 + s288);
    t576 = (s285 - s287);
    t577 = (s286 - s288);
    s289 = P1[(a979 + 16)];
    s290 = P1[(a979 + 17)];
    s291 = P1[(a979 + 80)];
    s292 = P1[(a979 + 81)];
    t578 = (s289 + s291);
    t579 = (s290 + s292);
    a980 = (0.70710678118654757*(s289 - s291));
    a981 = (0.70710678118654757*(s290 - s292));
    s293 = (a980 + a981);
    s294 = (a981 - a980);
    s295 = P1[(a979 + 32)];
    s296 = P1[(a979 + 33)];
    s297 = P1[(a979 + 96)];
    s298 = P1[(a979 + 97)];
    t580 = (s295 + s297);
    t581 = (s296 + s298);
    t582 = (s295 - s297);
    t583 = (s296 - s298);
    s299 = P1[(a979 + 48)];
    s300 = P1[(a979 + 49)];
    s301 = P1[(a979 + 112)];
    s302 = P1[(a979 + 113)];
    t584 = (s299 + s301);
    t585 = (s300 + s302);
    a982 = (0.70710678118654757*(s300 - s302));
    a983 = (0.70710678118654757*(s299 - s301));
    s303 = (a982 - a983);
    s304 = (a983 + a982);
    t586 = (t574 + t580);
    t587 = (t575 + t581);
    t588 = (t574 - t580);
    t589 = (t575 - t581);
    t590 = (t578 + t584);
    t591 = (t579 + t585);
    t592 = (t578 - t584);
    t593 = (t579 - t585);
    a984 = (a977 + (16*a978));
    T25[a984] = (t586 + t590);
    T25[(a984 + 1)] = (t587 + t591);
    T25[(a984 + 8)] = (t586 - t590);
    T25[(a984 + 9)] = (t587 - t591);
    T25[(a984 + 4)] = (t588 + t593);
    T25[(a984 + 5)] = (t589 - t592);
    T25[(a984 + 12)] = (t588 - t593);
    T25[(a984 + 13)] = (t589 + t592);
    t594 = (t576 + t583);
    t595 = (t577 - t582);
    t596 = (t576 - t583);
    t597 = (t577 + t582);
    t598 = (s293 + s303);
    t599 = (s294 - s304);
    t600 = (s293 - s303);
    t601 = (s294 + s304);
    T25[(a984 + 2)] = (t594 + t598);
    T25[(a984 + 3)] = (t595 + t599);
    T25[(a984 + 10)] = (t594 - t598);
    T25[(a984 + 11)] = (t595 - t599);
    T25[(a984 + 6)] = (t596 + t601);
    T25[(a984 + 7)] = (t597 - t600);
    T25[(a984 + 14)] = (t596 - t601);
    T25[(a984 + 15)] = (t597 + t600);
    __syncthreads();
    double a1463, a1464, a1465, a1466, a1467, a1468, a1469, a1470, 
            a1471, a1472, a1473, a1474, a1475, a1476, a1477, a1478, 
            a1479, a1480, a1481, a1482, s414, s415, s416, s417, 
            s418, s419, s420, s421, s422, s423, s424, s425, 
            s426, s427, s428, s429, s430, s431, s432, s433, 
            s434, s435, s436, s437, s438, s439, s440, s441, 
            s442, s443, s444, s445, s446, s447, s448, s449, 
            t726, t727, t728, t729, t730, t731, t732, t733, 
            t734, t735, t736, t737, t738, t739, t740, t741, 
            t742, t743, t744, t745, t746, t747, t748, t749, 
            t750, t751, t752, t753;
    int a1459, a1460, a1461, a1462, a1483;
    a1459 = (threadIdx.x / 8);
    a1460 = (threadIdx.x % 8);
    a1461 = ((128*a1459) + (2*a1460));
    s414 = T25[a1461];
    s415 = T25[(a1461 + 1)];
    s416 = T25[(a1461 + 64)];
    s417 = T25[(a1461 + 65)];
    a1462 = (16*a1460);
    a1463 = D3[a1462];
    a1464 = D3[(a1462 + 1)];
    s418 = ((a1463*s414) - (a1464*s415));
    s419 = ((a1464*s414) + (a1463*s415));
    a1465 = D3[(a1462 + 2)];
    a1466 = D3[(a1462 + 3)];
    s420 = ((a1465*s416) - (a1466*s417));
    s421 = ((a1466*s416) + (a1465*s417));
    t726 = (s418 + s420);
    t727 = (s419 + s421);
    t728 = (s418 - s420);
    t729 = (s419 - s421);
    s422 = T25[(a1461 + 16)];
    s423 = T25[(a1461 + 17)];
    s424 = T25[(a1461 + 80)];
    s425 = T25[(a1461 + 81)];
    a1467 = D3[(a1462 + 4)];
    a1468 = D3[(5 + a1462)];
    s426 = ((a1467*s422) - (a1468*s423));
    s427 = ((a1468*s422) + (a1467*s423));
    a1469 = D3[(6 + a1462)];
    a1470 = D3[(7 + a1462)];
    s428 = ((a1469*s424) - (a1470*s425));
    s429 = ((a1470*s424) + (a1469*s425));
    t730 = (s426 + s428);
    t731 = (s427 + s429);
    a1471 = (0.70710678118654757*(s426 - s428));
    a1472 = (0.70710678118654757*(s427 - s429));
    s430 = (a1471 + a1472);
    s431 = (a1472 - a1471);
    s432 = T25[(a1461 + 32)];
    s433 = T25[(a1461 + 33)];
    s434 = T25[(a1461 + 96)];
    s435 = T25[(a1461 + 97)];
    a1473 = D3[(a1462 + 8)];
    a1474 = D3[(9 + a1462)];
    s436 = ((a1473*s432) - (a1474*s433));
    s437 = ((a1474*s432) + (a1473*s433));
    a1475 = D3[(10 + a1462)];
    a1476 = D3[(11 + a1462)];
    s438 = ((a1475*s434) - (a1476*s435));
    s439 = ((a1476*s434) + (a1475*s435));
    t732 = (s436 + s438);
    t733 = (s437 + s439);
    t734 = (s436 - s438);
    t735 = (s437 - s439);
    s440 = T25[(a1461 + 48)];
    s441 = T25[(a1461 + 49)];
    s442 = T25[(a1461 + 112)];
    s443 = T25[(a1461 + 113)];
    a1477 = D3[(a1462 + 12)];
    a1478 = D3[(13 + a1462)];
    s444 = ((a1477*s440) - (a1478*s441));
    s445 = ((a1478*s440) + (a1477*s441));
    a1479 = D3[(14 + a1462)];
    a1480 = D3[(15 + a1462)];
    s446 = ((a1479*s442) - (a1480*s443));
    s447 = ((a1480*s442) + (a1479*s443));
    t736 = (s444 + s446);
    t737 = (s445 + s447);
    a1481 = (0.70710678118654757*(s445 - s447));
    a1482 = (0.70710678118654757*(s444 - s446));
    s448 = (a1481 - a1482);
    s449 = (a1482 + a1481);
    t738 = (t726 + t732);
    t739 = (t727 + t733);
    t740 = (t726 - t732);
    t741 = (t727 - t733);
    t742 = (t730 + t736);
    t743 = (t731 + t737);
    t744 = (t730 - t736);
    t745 = (t731 - t737);
    a1483 = ((32*blockIdx.x) + (8192*a1460) + (2*a1459));
    P2[a1483] = (t738 + t742);
    P2[(a1483 + 1)] = (t739 + t743);
    P2[(a1483 + 262144)] = (t738 - t742);
    P2[(a1483 + 262145)] = (t739 - t743);
    P2[(a1483 + 131072)] = (t740 + t745);
    P2[(a1483 + 131073)] = (t741 - t744);
    P2[(a1483 + 393216)] = (t740 - t745);
    P2[(a1483 + 393217)] = (t741 + t744);
    t746 = (t728 + t735);
    t747 = (t729 - t734);
    t748 = (t728 - t735);
    t749 = (t729 + t734);
    t750 = (s430 + s448);
    t751 = (s431 - s449);
    t752 = (s430 - s448);
    t753 = (s431 + s449);
    P2[(a1483 + 65536)] = (t746 + t750);
    P2[(a1483 + 65537)] = (t747 + t751);
    P2[(a1483 + 327680)] = (t746 - t750);
    P2[(a1483 + 327681)] = (t747 - t751);
    P2[(a1483 + 196608)] = (t748 + t753);
    P2[(a1483 + 196609)] = (t749 - t752);
    P2[(a1483 + 458752)] = (t748 - t753);
    P2[(a1483 + 458753)] = (t749 + t752);
    __syncthreads();
}
#endif

__global__ void LAUNCHBOUNDS ker_fftx_mddft_64x64x64_HIP2(double  *Y) {
    __shared__ double T47[2048];
    double a1721, a1722, a1723, a1724, s510, s511, s512, s513, 
            s514, s515, s516, s517, s518, s519, s520, s521, 
            s522, s523, s524, s525, s526, s527, s528, s529, 
            t878, t879, t880, t881, t882, t883, t884, t885, 
            t886, t887, t888, t889, t890, t891, t892, t893, 
            t894, t895, t896, t897, t898, t899, t900, t901, 
            t902, t903, t904, t905;
    int a1718, a1719, a1720, a1725;
//	for ( a1718 = 0; a1718 < 2048; a1718++ ) T47[ a1718 ] = 0.0;
#if 0
    a1718 = (128*(threadIdx.x / 8));
    a1719 = (threadIdx.x % 8);
#endif
    a1720 = ((2048*blockIdx.x) + a1718 + (2*a1719));
#if 0
    s510 = P2[a1720];
    s511 = P2[(a1720 + 1)];
    s512 = P2[(a1720 + 64)];
    s513 = P2[(a1720 + 65)];
    t878 = (s510 + s512);
    t879 = (s511 + s513);
    t880 = (s510 - s512);
    t881 = (s511 - s513);
    s514 = P2[(a1720 + 16)];
    s515 = P2[(a1720 + 17)];
    s516 = P2[(a1720 + 80)];
    s517 = P2[(a1720 + 81)];
    t882 = (s514 + s516);
    t883 = (s515 + s517);
    a1721 = (0.70710678118654757*(s514 - s516));
    a1722 = (0.70710678118654757*(s515 - s517));
    s518 = (a1721 + a1722);
    s519 = (a1722 - a1721);
    s520 = P2[(a1720 + 32)];
    s521 = P2[(a1720 + 33)];
    s522 = P2[(a1720 + 96)];
    s523 = P2[(a1720 + 97)];
    t884 = (s520 + s522);
    t885 = (s521 + s523);
    t886 = (s520 - s522);
    t887 = (s521 - s523);
    s524 = P2[(a1720 + 48)];
    s525 = P2[(a1720 + 49)];
    s526 = P2[(a1720 + 112)];
    s527 = P2[(a1720 + 113)];
    t888 = (s524 + s526);
    t889 = (s525 + s527);
    a1723 = (0.70710678118654757*(s525 - s527));
    a1724 = (0.70710678118654757*(s524 - s526));
    s528 = (a1723 - a1724);
    s529 = (a1724 + a1723);
    t890 = (t878 + t884);
    t891 = (t879 + t885);
    t892 = (t878 - t884);
    t893 = (t879 - t885);
    t894 = (t882 + t888);
    t895 = (t883 + t889);
    t896 = (t882 - t888);
    t897 = (t883 - t889);
    a1725 = (a1718 + (16*a1719));
    T47[a1725] = (t890 + t894);
    T47[(a1725 + 1)] = (t891 + t895);
    T47[(a1725 + 8)] = (t890 - t894);
    T47[(a1725 + 9)] = (t891 - t895);
    T47[(a1725 + 4)] = (t892 + t897);
    T47[(a1725 + 5)] = (t893 - t896);
    T47[(a1725 + 12)] = (t892 - t897);
    T47[(a1725 + 13)] = (t893 + t896);
    t898 = (t880 + t887);
    t899 = (t881 - t886);
    t900 = (t880 - t887);
    t901 = (t881 + t886);
    t902 = (s518 + s528);
    t903 = (s519 - s529);
    t904 = (s518 - s528);
    t905 = (s519 + s529);
    T47[(a1725 + 2)] = (t898 + t902);
    T47[(a1725 + 3)] = (t899 + t903);
    T47[(a1725 + 10)] = (t898 - t902);
    T47[(a1725 + 11)] = (t899 - t903);
    T47[(a1725 + 6)] = (t900 + t905);
    T47[(a1725 + 7)] = (t901 - t904);
    T47[(a1725 + 14)] = (t900 - t905);
    T47[(a1725 + 15)] = (t901 + t904);
    __syncthreads();
#endif
    double a2204, a2205, a2206, a2207, a2208, a2209, a2210, a2211, 
            a2212, a2213, a2214, a2215, a2216, a2217, a2218, a2219, 
            a2220, a2221, a2222, a2223, s638, s639, s640, s641, 
            s642, s643, s644, s645, s646, s647, s648, s649, 
            s650, s651, s652, s653, s654, s655, s656, s657, 
            s658, s659, s660, s661, s662, s663, s664, s665, 
            s666, s667, s668, s669, s670, s671, s672, s673, 
            t1030, t1031, t1032, t1033, t1034, t1035, t1036, t1037, 
            t1038, t1039, t1040, t1041, t1042, t1043, t1044, t1045, 
            t1046, t1047, t1048, t1049, t1050, t1051, t1052, t1053, 
            t1054, t1055, t1056, t1057;
    int a2200, a2201, a2202, a2203, a2224;
    a2200 = (threadIdx.x / 8);
    a2201 = (threadIdx.x % 8);
    a2202 = ((128*a2200) + (2*a2201));
    s638 = T47[a2202];
    s639 = T47[(a2202 + 1)];
    s640 = T47[(a2202 + 64)];
    s641 = T47[(a2202 + 65)];
    a2203 = (16*a2201);
    a2204 = D3[a2203];
    a2205 = D3[(a2203 + 1)];
    s642 = ((a2204*s638) - (a2205*s639));
    s643 = ((a2205*s638) + (a2204*s639));
    a2206 = D3[(a2203 + 2)];
    a2207 = D3[(a2203 + 3)];
    s644 = ((a2206*s640) - (a2207*s641));
    s645 = ((a2207*s640) + (a2206*s641));
    t1030 = (s642 + s644);
    t1031 = (s643 + s645);
    t1032 = (s642 - s644);
    t1033 = (s643 - s645);
    s646 = T47[(a2202 + 16)];
    s647 = T47[(a2202 + 17)];
    s648 = T47[(a2202 + 80)];
    s649 = T47[(a2202 + 81)];
    a2208 = D3[(a2203 + 4)];
    a2209 = D3[(5 + a2203)];
    s650 = ((a2208*s646) - (a2209*s647));
    s651 = ((a2209*s646) + (a2208*s647));
    a2210 = D3[(6 + a2203)];
    a2211 = D3[(7 + a2203)];
    s652 = ((a2210*s648) - (a2211*s649));
    s653 = ((a2211*s648) + (a2210*s649));
    t1034 = (s650 + s652);
    t1035 = (s651 + s653);
    a2212 = (0.70710678118654757*(s650 - s652));
    a2213 = (0.70710678118654757*(s651 - s653));
    s654 = (a2212 + a2213);
    s655 = (a2213 - a2212);
    s656 = T47[(a2202 + 32)];
    s657 = T47[(a2202 + 33)];
    s658 = T47[(a2202 + 96)];
    s659 = T47[(a2202 + 97)];
    a2214 = D3[(a2203 + 8)];
    a2215 = D3[(9 + a2203)];
    s660 = ((a2214*s656) - (a2215*s657));
    s661 = ((a2215*s656) + (a2214*s657));
    a2216 = D3[(10 + a2203)];
    a2217 = D3[(11 + a2203)];
    s662 = ((a2216*s658) - (a2217*s659));
    s663 = ((a2217*s658) + (a2216*s659));
    t1036 = (s660 + s662);
    t1037 = (s661 + s663);
    t1038 = (s660 - s662);
    t1039 = (s661 - s663);
    s664 = T47[(a2202 + 48)];
    s665 = T47[(a2202 + 49)];
    s666 = T47[(a2202 + 112)];
    s667 = T47[(a2202 + 113)];
    a2218 = D3[(a2203 + 12)];
    a2219 = D3[(13 + a2203)];
    s668 = ((a2218*s664) - (a2219*s665));
    s669 = ((a2219*s664) + (a2218*s665));
    a2220 = D3[(14 + a2203)];
    a2221 = D3[(15 + a2203)];
    s670 = ((a2220*s666) - (a2221*s667));
    s671 = ((a2221*s666) + (a2220*s667));
    t1040 = (s668 + s670);
    t1041 = (s669 + s671);
    a2222 = (0.70710678118654757*(s669 - s671));
    a2223 = (0.70710678118654757*(s668 - s670));
    s672 = (a2222 - a2223);
    s673 = (a2223 + a2222);
    t1042 = (t1030 + t1036);
    t1043 = (t1031 + t1037);
    t1044 = (t1030 - t1036);
    t1045 = (t1031 - t1037);
    t1046 = (t1034 + t1040);
    t1047 = (t1035 + t1041);
    t1048 = (t1034 - t1040);
    t1049 = (t1035 - t1041);
    a2224 = ((32*blockIdx.x) + (8192*a2201) + (2*a2200));
    Y[a2224] = (t1042 + t1046);
    Y[(a2224 + 1)] = (t1043 + t1047);
    Y[(a2224 + 262144)] = (t1042 - t1046);
    Y[(a2224 + 262145)] = (t1043 - t1047);
    Y[(a2224 + 131072)] = (t1044 + t1049);
    Y[(a2224 + 131073)] = (t1045 - t1048);
    Y[(a2224 + 393216)] = (t1044 - t1049);
    Y[(a2224 + 393217)] = (t1045 + t1048);
    t1050 = (t1032 + t1039);
    t1051 = (t1033 - t1038);
    t1052 = (t1032 - t1039);
    t1053 = (t1033 + t1038);
    t1054 = (s654 + s672);
    t1055 = (s655 - s673);
    t1056 = (s654 - s672);
    t1057 = (s655 + s673);
    Y[(a2224 + 65536)] = (t1050 + t1054);
    Y[(a2224 + 65537)] = (t1051 + t1055);
    Y[(a2224 + 327680)] = (t1050 - t1054);
    Y[(a2224 + 327681)] = (t1051 - t1055);
    Y[(a2224 + 196608)] = (t1052 + t1057);
    Y[(a2224 + 196609)] = (t1053 - t1056);
    Y[(a2224 + 458752)] = (t1052 - t1057);
    Y[(a2224 + 458753)] = (t1053 + t1056);
    __syncthreads();
}

extern "C" {
void fftx_mddft_64x64x64_HIP(double  *Y, double  *X, double  *sym) {
    dim3 b283(128, 1, 1), b284(128, 1, 1), b285(128, 1, 1), g1(256, 1, 1), g2(256, 1, 1), g3(256, 1, 1);
//    hipLaunchKernelGGL(ker_fftx_mddft_64x64x64_HIP0, dim3(g1), dim3(b283), 0, 0, X);
//    hipLaunchKernelGGL(ker_fftx_mddft_64x64x64_HIP1, dim3(g2), dim3(b284), 0, 0);
    hipLaunchKernelGGL(ker_fftx_mddft_64x64x64_HIP2, dim3(g3), dim3(b285), 0, 0, Y);
}
}

extern "C" {
void destroy_fftx_mddft_64x64x64_HIP() {
    double  *hp1;
    hipMemcpyFromSymbol(&(hp1), HIP_SYMBOL(P1), sizeof(double  *));
    hipFree(hp1);
    hipMemcpyFromSymbol(&(hp1), HIP_SYMBOL(P2), sizeof(double  *));
    hipFree(hp1);
}
}

extern "C" {
void init_fftx_mddft_64x64x64_HIP() {
    double  *hp1;
//    hipFuncSetCacheConfig(reinterpret_cast<const void*>(ker_fftx_mddft_64x64x64_HIP0), hipFuncCachePreferL1);
//    hipFuncSetCacheConfig(reinterpret_cast<const void*>(ker_fftx_mddft_64x64x64_HIP1), hipFuncCachePreferL1);
    hipFuncSetCacheConfig(reinterpret_cast<const void*>(ker_fftx_mddft_64x64x64_HIP2), hipFuncCachePreferL1);
    hipMalloc(((void  * *) &(hp1)), (sizeof(double )*524288));
    hipMemcpyToSymbol(HIP_SYMBOL(P1), &(hp1), sizeof(double  *));
    hipMalloc(((void  * *) &(hp1)), (sizeof(double )*524288));
    hipMemcpyToSymbol(HIP_SYMBOL(P2), &(hp1), sizeof(double  *));
}
}
