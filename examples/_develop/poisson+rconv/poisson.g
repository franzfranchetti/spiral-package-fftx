#=============================================================================
# Real Free Space Convolution

ImportAll(realdft);
ImportAll(filtering);
ImportAll(dct_dst);
ImportAll(fftx.nonterms);

n := 4;

idx := Cartesian([0..n-1]-n/2, [0..n/2]-n/2);
#idx := Cartesian([0..n-1]-n/2, [0..n/2]-n/2);
symbf := List(idx, i->let(v := i[1]^2*i[2]^2, When(v=0, 0, 1/v)));
dft := MDPRDFT([n,n], -1);
idft := 1/n^2 * IMDPRDFT([n,n], -1);
MatSPL(idft * dft);
diag := RC(Diag(symbf));

conv := idft * diag * dft;
convm := MatSPL(conv);

convmi := InfinityNormMat(List(convm, r->List(r, v->Im(v))));
convmr := List(convm, r->List(r, v->Re(v)));

pm(convmr);

# find the filters
am := [[var("a"), var("b")], [var("c"), var("d")]]; 
#am := [[1,2], [3,4]]; 
tm := ApplyFunc(VStack, List(am, rr->ApplyFunc(HStack, List(rr, i-> Blk(List(am, r->List(r, c-> i*c)))))));


flt := Tensor(am, am);
