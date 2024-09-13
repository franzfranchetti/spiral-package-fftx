ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);

SkewCirculant := l -> Toeplitz(Reversed(l)::(-DropLast(Reversed(l), 1)));

m := 2;
n := 2*m;
range := [1..n];
filt := List(range, i->Random([1..100]));

# circulant case
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
