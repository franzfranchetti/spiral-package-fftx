# HProduct(<list>) - scaled Hadamard Product y = αx * βz
Class(HProduct, RowVec);

#might not need this RowVec doesn't have this I believe
DefaultSumsGen.HProduct := (self, o, opts) >> o;

DefaultCodegen.HProduct := (self, o, y, x, opts) >> let(i := Ind(), func := o.element.lambda(),
        t := TempVar(x.t.t),
        chain(assign(t,1),
            loop(i, func.domain(), assign(t, mul(t, mul(func.at(i), nth(x,i))))),
            assign(nth(y,0), t)));
