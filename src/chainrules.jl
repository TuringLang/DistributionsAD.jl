## Uniform ##

@scalar_rule(
    uniformlogpdf(a, b, x),
    @setup(
        insupport = a <= x <= b,
        diff = b - a,
        c = insupport ? inv(diff) : inv(one(diff)),
    ),
    (c, -c, ZeroTangent()),
)
