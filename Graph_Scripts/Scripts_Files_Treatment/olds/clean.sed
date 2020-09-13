%s/plot.*//g
%g/set t.*/d
%s/^e.*//g
%g/set y.*/d
%g/set x.*/d
%g/^$/d
%s/set.*run_\([0-9]\{1,2\}\)\..*/\rRUN \1/g

