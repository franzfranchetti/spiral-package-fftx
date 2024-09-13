# 2-way parallel 1D real convolution
#
# break free space convolution -> circulant + skew-circulant
# break both circulant and skew-circulant into 2x2 Toeplitz
# do 2 Toeplitz on proc0, and 2 on proc1
# either broadcast all data or reduce+ at the end
# 2x n/2*sizeof(real) data packets transmitted, rest independent

ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));

# set up problem
m := 8;
n := 2*m;
range := [1..n];
filt := List(range, i->Random([1..100]));

# Circulant case --------------------------------------------------------------
conv := Circulant(filt);
convm := MatSPL(conv);
pm(convm);

conv00 := Gath(fTensor(fBase(2,0), fId(m))) * conv * Scat(fTensor(fBase(2,0), fId(m)));
conv01 := Gath(fTensor(fBase(2,0), fId(m))) * conv * Scat(fTensor(fBase(2,1), fId(m)));
conv10 := Gath(fTensor(fBase(2,1), fId(m))) * conv * Scat(fTensor(fBase(2,0), fId(m)));
conv11 := Gath(fTensor(fBase(2,1), fId(m))) * conv * Scat(fTensor(fBase(2,1), fId(m)));

pm(conv00);
pm(conv01);
pm(conv10);
pm(conv11);

conv2 := SUM(
    Scat(fTensor(fBase(2,0), fId(m))) * conv00 * Gath(fTensor(fBase(2,0), fId(m))), 
    Scat(fTensor(fBase(2,0), fId(m))) * conv01 * Gath(fTensor(fBase(2,1), fId(m))), 
    Scat(fTensor(fBase(2,1), fId(m))) * conv10 * Gath(fTensor(fBase(2,0), fId(m))), 
    Scat(fTensor(fBase(2,1), fId(m))) * conv11 * Gath(fTensor(fBase(2,1), fId(m))));
InfinityNormMat(MatSPL(conv2) - convm);

conv00t := Toeplitz(Reversed(filt{[2..m]})::filt{[1]}::Reversed(filt{[m+2..2*m]}));
conv11t := conv00t;
conv01t := Toeplitz(Reversed(filt{[2..2*m]}));
conv10t := Toeplitz(Reversed(filt{[m+2..2*m]})::filt{[m+1]}::Reversed(filt{[2..m]}));

pm(conv00t);
pm(conv01t);
pm(conv10t);
pm(conv11t);

InfinityNormMat(MatSPL(conv00) - MatSPL(conv00t));
InfinityNormMat(MatSPL(conv01) - MatSPL(conv01t));
InfinityNormMat(MatSPL(conv10) - MatSPL(conv10t));
InfinityNormMat(MatSPL(conv11) - MatSPL(conv11t));

pconv1 := Scat(fTensor(fBase(2,0), fId(m))) * SUM(conv00t * Gath(fTensor(fBase(2,0), fId(m))), conv01t * Gath(fTensor(fBase(2,1), fId(m))));
pconv2 := Scat(fTensor(fBase(2,1), fId(m))) * SUM(conv10t * Gath(fTensor(fBase(2,0), fId(m))), conv11t * Gath(fTensor(fBase(2,1), fId(m))));

pconv := SUM(pconv1, pconv2);
InfinityNormMat(MatSPL(pconv) - convm);

# Skew-Circulant case ---------------------------------------------------------
sconv := SkewCirculant(filt);
sconvm := MatSPL(sconv);
pm(sconvm);

sconv00 := Gath(fTensor(fBase(2,0), fId(m))) * sconv * Scat(fTensor(fBase(2,0), fId(m)));
sconv01 := Gath(fTensor(fBase(2,0), fId(m))) * sconv * Scat(fTensor(fBase(2,1), fId(m)));
sconv10 := Gath(fTensor(fBase(2,1), fId(m))) * sconv * Scat(fTensor(fBase(2,0), fId(m)));
sconv11 := Gath(fTensor(fBase(2,1), fId(m))) * sconv * Scat(fTensor(fBase(2,1), fId(m)));

pm(sconv00);
pm(sconv01);
pm(sconv10);
pm(sconv11);

sconv2 := SUM(
    Scat(fTensor(fBase(2,0), fId(m))) * sconv00 * Gath(fTensor(fBase(2,0), fId(m))), 
    Scat(fTensor(fBase(2,0), fId(m))) * sconv01 * Gath(fTensor(fBase(2,1), fId(m))), 
    Scat(fTensor(fBase(2,1), fId(m))) * sconv10 * Gath(fTensor(fBase(2,0), fId(m))), 
    Scat(fTensor(fBase(2,1), fId(m))) * sconv11 * Gath(fTensor(fBase(2,1), fId(m))));
InfinityNormMat(MatSPL(sconv2) - sconvm);

sconv00t := Toeplitz(Reversed(filt{[2..m]})::filt{[1]}::Reversed(-filt{[m+2..2*m]}));
sconv11t := sconv00t;
sconv01t := Toeplitz(Reversed(filt{[2..2*m]}));
sconv10t := Toeplitz(-(Reversed(filt{[m+2..2*m]})::filt{[m+1]}::Reversed(filt{[2..m]})));

pm(sconv00t);
pm(sconv01t);
pm(sconv10t);
pm(sconv11t);

InfinityNormMat(MatSPL(sconv00) - MatSPL(sconv00t));
InfinityNormMat(MatSPL(sconv01) - MatSPL(sconv01t));
InfinityNormMat(MatSPL(conv10) - MatSPL(conv10t));
InfinityNormMat(MatSPL(sconv11) - MatSPL(sconv11t));

psconv1 := Scat(fTensor(fBase(2,0), fId(m))) * SUM(sconv00t * Gath(fTensor(fBase(2,0), fId(m))), sconv01t * Gath(fTensor(fBase(2,1), fId(m))));
psconv2 := Scat(fTensor(fBase(2,1), fId(m))) * SUM(sconv10t * Gath(fTensor(fBase(2,0), fId(m))), sconv11t * Gath(fTensor(fBase(2,1), fId(m))));

psconv := SUM(psconv1, psconv2);
InfinityNormMat(MatSPL(psconv) - sconvm);

