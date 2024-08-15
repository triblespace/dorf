import marimo

__generated_with = "0.6.26"
app = marimo.App()


@app.cell
def __():
    import maturin_import_hook
    maturin_import_hook.install(settings=maturin_import_hook.MaturinSettings(release=True))
    return maturin_import_hook,


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell
def __():
    import polars as pl
    return pl,


@app.cell
def __():
    import os
    return os,


@app.cell
def __():
    import hifitime
    return hifitime,


@app.cell
def __():
    import time
    return time,


@app.cell
def __():
    import timeit
    return timeit,


@app.cell
def __():
    import fractions
    from fractions import Fraction
    return Fraction, fractions


@app.cell
def __():
    import altair as alt
    return alt,


@app.cell
def __():
    import dorf
    import importlib
    importlib.reload(dorf)
    from dorf import bench
    from dorf import tribles
    return bench, dorf, importlib, tribles


@app.cell
def __(mo):
    mo.md("# Tribles")
    return


@app.cell
def __(tribles):
    TribleSet = tribles.TribleSet
    return TribleSet,


@app.cell
def __(tribles):
    Id = tribles.Id
    return Id,


@app.cell
def __(tribles):
    Value = tribles.Value
    return Value,


@app.cell
def __(name):
    class Variable:
        def __init__(self, context, index, name=None):
            self.context = context
            self.index = index
            self.name = name
            self.schema = None

        def annotate_schema(self, schema):
            if self.schema is None:
                self.schema = schema
            else:
                if self.schema != schema:
                    raise TypeError(
                        "variable"
                        + name
                        + " annotated with conflicting schemas"
                        + str(self.schema)
                        + " and "
                        + str(schema)
                    )
    return Variable,


@app.cell
def __(Id, RndId, Value, Variable, tribles):
    class Namespace:
        def __init__(self, declaration):
            self.declaration = declaration

        def entity(self, entity):
            set = tribles.TribleSet.empty()
            if Id in entity:
                entity_id = entity[Id]
            else:
                entity_id = Id.genid()

            for key, value in entity.items():
                if key is Id:
                    continue
                attr_id = self.declaration[key][1]
                attr_schema = self.declaration[key][0]
                value = Value.of(attr_schema, value)
                set.add(entity_id, attr_id, value)

            return set

        def pattern(self, ctx, set, entities):
            constraints = []
            for entity in entities:
                if Id in entity:
                    entity_id = entity[Id]
                else:
                    entity_id = ctx.new()
                if type(entity_id) is Variable:
                    e_v = entity_id
                    e_v.annotate_schema(RndId)
                else:
                    e_v = ctx.new()
                    e_v.annotate_schema(RndId)
                    constraints.append(
                        tribles.constant(
                            e_v.index,
                            Value.of(RndId, entity_id),
                    ))
        
                for key, value in entity.items():
                    if key is Id:
                        continue
                    attr_id = self.declaration[key][1]
                    attr_schema = self.declaration[key][0]
                    
                    a_v = ctx.new()
                    a_v.annotate_schema(RndId)
                    constraints.append(
                        tribles.constant(
                            a_v.index,
                            Value.of(RndId, attr_id),
                    ))

                    if type(value) is Variable:
                        v_v = value
                        v_v.annotate_schema(attr_schema)
                    else:
                        v_v = ctx.new()
                        v_v.annotate_schema(attr_schema)
                        constraints.append(
                            tribles.constant(
                                v_v.index,
                                Value.of(attr_schema, value)
                        ))
                    constraints.append(set.pattern(e_v.index, a_v.index, v_v.index))
            return tribles.intersect(constraints)
    return Namespace,


@app.cell
def __(Namespace):
    def ns(declaration):
        return Namespace(declaration)
    return ns,


@app.cell
def __(Variable):
    class VariableContext:
        def __init__(self):
            self.variables = []

        def new(self, name=None):
            i = len(self.variables)
            assert i < 128
            v = Variable(self, i, name)
            self.variables.append(v)
            return v

        def check_schemas(self):
            for v in self.variables:
                if not v.schema:
                    if v.name:
                        name = "'" + v.name + "'"
                    else:
                        name = "_"
                    raise TypeError(
                        "missing schema for variable "
                        + name
                        + "/"
                        + str(v.index)
                    )
    return VariableContext,


@app.cell
def __(VariableContext, tribles):
    def find(query):
        ctx = VariableContext()
        projected_variable_names = query.__code__.co_varnames[1:]
        projected_variables = [ctx.new(n) for n in projected_variable_names]
        constraint = query(ctx, *projected_variables)
        ctx.check_schemas()
        projected_variable_schemas = [(v.index, v.schema) for v in projected_variables]
        for result in tribles.solve(projected_variable_schemas, constraint):
            yield tuple(result)
    return find,


@app.cell
def __(tribles):
    def register_type(entity_id):
        def inner(type):
            tribles.register_type(entity_id, type)
            return type
        return inner
    return register_type,


@app.cell
def __(tribles):
    def register_converter(schema, type):
        def inner(converter):
            tribles.register_converter(schema, type, converter)
            return converter
        return inner
    return register_converter,


@app.cell
def __(Id, fractions, register_type):
    register_type(Id.hex("A75056BFA2AE677767B1DB8B01AFA322"))(int)
    register_type(Id.hex("7D06820D69947D76E7177E5DEA4EA773"))(str)
    register_type(Id.hex("BF11820EC384447B666988490D727A1C"))(Id)
    register_type(Id.hex("83D62F300ED37850DFFB42E6226117ED"))(fractions.Fraction)
    return


@app.cell
def __(Id):
    """an random 128 bit id (the first 128bits are zero padding)"""
    RndId = Id.hex("DFA138FA94D059161C9AB8C800F6FEC4")
    return RndId,


@app.cell
def __(Id):
    """32 raw bytes"""
    RawBytes = Id.hex("7B374E233C226597E8C7D8C6215504F0")
    return RawBytes,


@app.cell
def __(Id):
    """A \0 terminated short utf-8 string that fits of up to 32 bytes of characters"""
    ShortString = Id.hex("BDDBE1EDBCD3EF7B74CEB109DE67A47B")
    return ShortString,


@app.cell
def __(Id):
    """an signed 256bit integer in big endian encoding"""
    I256BE = Id.hex("5F80F30E596C2CEF2AFDDFCBD9933AC7")
    return I256BE,


@app.cell
def __(Id):
    """a signed 256bit integer in little endian encoding"""
    I256LE = Id.hex("F5E93737BFD910EDE8902ACAA8493CEE")
    return I256LE,


@app.cell
def __(Id):
    """an unsigned 256bit integer in big endian encoding"""
    U256BE = Id.hex("5E868BA4B9C06DD12E7F4AA064D1A7C7")
    return U256BE,


@app.cell
def __(Id):
    """an unsigned 256bit integer in little endian encoding"""
    U256LE = Id.hex("EC9C2F8C3C3156BD203D92888D7479CD")
    return U256LE,


@app.cell
def __(Id):
    """a time duration in nanoseconds stored as a signed 256bit big endian integer"""
    NSDuration = Id.hex("BD1DA74AABF1D01A5CF4EEF3683B1EC5")
    return NSDuration,


@app.cell
def __(Id):
    """a unitless fraction stored as a (numerator, denominator) pair of signed 128bit little endian integers"""
    FR256LE = Id.hex("77694E74654A039625FA5911381F3897")
    return FR256LE,


@app.cell
def __(RndId, register_converter):
    @register_converter(schema = RndId, type = str)
    class RndId_str_Converter:
        @staticmethod
        def pack(value):
            assert len(value) == 32
            return bytes.fromhex(value)
        @staticmethod
        def unpack(bytes):
            return bytes.hex().upper()
    return RndId_str_Converter,


@app.cell
def __(Id, RndId, register_converter):
    @register_converter(schema = RndId, type = Id)
    class RndId_Id_Converter:
        @staticmethod
        def pack(value):
            return bytes(16) + value.bytes()
        @staticmethod
        def unpack(bytes):
            assert all(v == 0 for v in bytes[0: 16])
            assert not all(v == 0 for v in bytes[16: 32])
            return Id(bytes[16:32])
    return RndId_Id_Converter,


@app.cell
def __(ShortString, register_converter):
    @register_converter(schema = ShortString, type = str)
    class ShortString_str_Converter:
        @staticmethod
        def pack(value):
            b = bytes(value, 'utf-8')
            assert len(b) <= 32
            assert 0 not in b
            return b + bytes(32 - len(b))
        @staticmethod
        def unpack(bytes):
            try:
                end = bytes.index(0)
                return bytes[0:end].decode('utf-8')
            except:
                return bytes.decode('utf-8')

    return ShortString_str_Converter,


@app.cell
def __(I256BE, register_converter):
    @register_converter(schema = I256BE, type = int)
    class I256BE_Int_Converter:
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=True)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=True)
    return I256BE_Int_Converter,


@app.cell
def __(I256LE, register_converter):
    @register_converter(schema = I256LE, type = int)
    class I256LE_Int_Converter:
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='little', signed=True)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='little', signed=True)
    return I256LE_Int_Converter,


@app.cell
def __(U256BE, register_converter):
    @register_converter(schema = U256BE, type = int)
    class U256BE_Int_Converter:
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=False)
    return U256BE_Int_Converter,


@app.cell
def __(U256LE, register_converter):
    @register_converter(schema = U256LE, type = int)
    class U256LE_Int_Converter:
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='little', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='little', signed=False)
    return U256LE_Int_Converter,


@app.cell
def __(NSDuration, register_converter):
    @register_converter(schema = NSDuration, type = int)
    class NSDuration_Int_Converter:
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=False)
    return NSDuration_Int_Converter,


@app.cell
def __(FR256LE, fractions, register_converter):
    @register_converter(schema = FR256LE, type = fractions.Fraction)
    class FR128LE_Fraction_Converter:
        @staticmethod
        def pack(value):
            n, d = value.as_integer_ratio()
            nb = n.to_bytes(16, byteorder='little', signed=True)
            db = d.to_bytes(16, byteorder='little', signed=True)
            return nb + db
        @staticmethod
        def unpack(bytes):
            n = int.from_bytes(bytes[0:16], byteorder='little', signed=True)
            d = int.from_bytes(bytes[16:32], byteorder='little', signed=True)
            return fractions.Fraction(n, d)
    return FR128LE_Fraction_Converter,


@app.cell
def __(FR256LE, Id, NSDuration, RndId, ShortString, U256LE, ns):
    experiments = ns({
        "label": (ShortString, Id.hex("EC80E5FBDF856CD47347D1BCFB5E0D3E")),
        "experiment": (RndId, Id.hex("E3ABE180BD5742D92616671E643FA4E5")),
        "element_count": (U256LE, Id.hex("A8034B8D0D644DCAA053CA1374AE92A0")),
        "cpu_time": (NSDuration, Id.hex("1C333940F98D0CFCEBFCC408FA35FF92")),
        "wall_time": (NSDuration, Id.hex("999BF50FFECF9C0B62FD23689A6CA0D0")),
        "avg_distance": (FR256LE, Id.hex("78D9B9230C044FA4E1585AFD14CFB3EE")),
        "change_count": (U256LE, Id.hex("AD5DD3F72FA8DD67AF0D0DA5298A98B9")),
        "layer_explored": (U256LE, Id.hex("2DB0F43553543173C42C8AE1573A38DB")),
    })
    return experiments,


@app.cell
def __(mo):
    mo.md("# Benchmarks")
    return


@app.cell
def __(Fraction, experiments):
    def gen_data(size):
        for i in range(size):
            yield experiments.entity({
                "layer_explored": i,
                "cpu_time": 500 * i,
                "wall_time": 600 * i,
                "avg_distance": Fraction(i, 1)})
    return gen_data,


@app.cell
def __(TribleSet, gen_data):
    def bench_consume(size):
        set = TribleSet.empty()
        for entity in gen_data(size):
            set.consume(entity)
        return set
    return bench_consume,


@app.cell
def __(TribleSet, gen_data):
    def bench_mutable_add(size):
        set = TribleSet.empty()
        for entity in gen_data(size):
            set += entity
        return set
    return bench_mutable_add,


@app.cell
def __(TribleSet, gen_data):
    def bench_sum(size):
        set = sum(gen_data(size), start = TribleSet.empty())
        return set
    return bench_sum,


@app.cell
def __(timeit):
    def time_ns(l):
        s = timeit.timeit(l, number=1)
        return int(s * 1e9)
    return time_ns,


@app.cell
def __(mo):
    mo.md("### Insert")
    return


@app.cell
def __(Id, bench_consume, element_count_exp, experiments, time_ns):
    _experiment = Id.genid()
    bench_insert_consume_data = experiments.entity({Id: _experiment, "label": "consume"})
    for _i in range(element_count_exp):
        bench_insert_consume_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_consume(2**_i)),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_insert_consume_data,


@app.cell
def __(Id, bench_mutable_add, element_count_exp, experiments, time_ns):
    _experiment = Id.genid()
    bench_insert_mutable_add_data = experiments.entity(
        {Id: _experiment, "label": "mutable_add"}
    )
    for _i in range(element_count_exp):
        bench_insert_mutable_add_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_mutable_add(2**_i)),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_insert_mutable_add_data,


@app.cell
def __(Id, bench_sum, element_count_exp, experiments, time_ns):
    _experiment = Id.genid()
    bench_insert_sum_data = experiments.entity({Id: _experiment, "label": "sum"})
    for _i in range(element_count_exp):
        bench_insert_sum_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_sum(2**_i)),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_insert_sum_data,


@app.cell
def __(mo):
    mo.md("### Query")
    return


@app.cell
def __(bench_consume, experiments, find, time_ns):
    def bench_query_find(size):
        set = bench_consume(size)
        return time_ns(
            lambda: sum(
                [
                    1
                    for _ in find(
                        lambda ctx, layer, cpu, wall, distance: experiments.pattern(
                            ctx,
                            set,
                            [
                                {
                                    "layer_explored": layer,
                                    "cpu_time": cpu,
                                    "wall_time": wall,
                                    "avg_distance": distance,
                                }
                            ],
                        )
                    )
                ]
            )
        )
    return bench_query_find,


@app.cell
def __(Id, bench_query_find, element_count_exp, experiments):
    _experiment = Id.genid()
    bench_query_find_data = experiments.entity({Id: _experiment, "label": "find"})
    for _i in range(element_count_exp):
        bench_query_find_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": bench_query_find(2**_i),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_query_find_data,


@app.cell
def __(mo):
    mo.md("## RDFLib")
    return


@app.cell
def __():
    from rdflib import Graph, URIRef, BNode, Literal, Namespace as RDFNamespace
    return BNode, Graph, Literal, RDFNamespace, URIRef


@app.cell
def __():
    from rdflib.plugins import sparql
    return sparql,


@app.cell
def __(RDFNamespace):
    benchns = RDFNamespace("http://example.org/benchmark/")
    rdf_layer_explored = benchns.layer_explored
    rdf_avg_cpu_time = benchns.avg_cpu_time
    rdf_avg_wall_time = benchns.avg_wall_time
    rdf_avg_distance = benchns.avg_distance
    return (
        benchns,
        rdf_avg_cpu_time,
        rdf_avg_distance,
        rdf_avg_wall_time,
        rdf_layer_explored,
    )


@app.cell
def __(mo):
    mo.md("### Insert")
    return


@app.cell
def __(BNode, Fraction, Graph, Literal, benchns):
    def bench_rdf(n):
        g = Graph()
        g.bind("benchmark", benchns)

        for i in range(n):
            eid = BNode()  # a GUID is generated
            g.add((eid, benchns.layer_explored, Literal(i)))
            g.add((eid, benchns.avg_cpu_time, Literal(500 * i)))
            g.add((eid, benchns.avg_wall_time, Literal(600 * i)))
            g.add((eid, benchns.avg_distance, Literal(Fraction(i, 1))))

        return g
    return bench_rdf,


@app.cell
def __(Id, bench_rdf, element_count_exp, experiments, time_ns):
    _experiment = Id.genid()
    bench_insert_rdf_data = experiments.entity({Id: _experiment, "label": "RDFLib"})
    for _i in range(element_count_exp):
        bench_insert_rdf_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": time_ns(lambda: bench_rdf(2**_i)),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_insert_rdf_data,


@app.cell
def __(mo):
    mo.md("### Query")
    return


@app.cell
def __(bench_rdf, time_ns):
    def bench_rdf_query_adhoc(n):
        g = bench_rdf(n)

        query = """
        SELECT ?layer ?cpu ?wall ?distance
        WHERE {
            ?a benchmark:layer_explored ?layer;
               benchmark:avg_cpu_time ?cpu;
               benchmark:avg_wall_time ?wall;
               benchmark:avg_distance ?distance .
        }"""
        return time_ns(lambda: sum([1 for _ in g.query(query)]))
    return bench_rdf_query_adhoc,


@app.cell
def __(bench_rdf, benchns, sparql, time_ns):
    _prepared_query = sparql.prepareQuery(
        """
        SELECT ?layer ?cpu ?wall ?distance
        WHERE {
            ?a benchmark:layer_explored ?layer;
               benchmark:avg_cpu_time ?cpu;
               benchmark:avg_wall_time ?wall;
               benchmark:avg_distance ?distance .
        }""",
        initNs = { "benchmark": benchns })

    def bench_rdf_query_prepared(n):
        g = bench_rdf(n) 
        return time_ns(lambda: sum([1 for _ in g.query(_prepared_query)]))
    return bench_rdf_query_prepared,


@app.cell
def __(Id, bench_rdf_query_adhoc, element_count_exp, experiments):
    _experiment = Id.genid()
    bench_query_adhoc_rdf_data = experiments.entity({Id: _experiment, "label": "RDFLib (adhoc)"})
    for _i in range(element_count_exp):
        bench_query_adhoc_rdf_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": bench_rdf_query_adhoc(2**_i),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_query_adhoc_rdf_data,


@app.cell
def __(Id, bench_rdf_query_prepared, element_count_exp, experiments):
    _experiment = Id.genid()
    bench_query_prepared_rdf_data = experiments.entity({Id: _experiment, "label": "RDFLib (prepared)"})
    for _i in range(element_count_exp):
        bench_query_prepared_rdf_data += experiments.entity(
            {
                "experiment": _experiment,
                "wall_time": bench_rdf_query_prepared(2**_i),
                "element_count": (2**_i) * 4,
            }
        )
    return bench_query_prepared_rdf_data,


@app.cell
def __(mo):
    mo.md("## Polars")
    return


@app.cell
def __(pl):
    df_triples = pl.DataFrame(
        {
            "e": ["a", "b", "c"],
            "a": [1, 2, 2],
            "v": [100, 200, 300],
        }
    )
    df_triples
    return df_triples,


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md("## Evaluation")
    return


@app.cell
def __():
    element_count_exp = 20
    return element_count_exp,


@app.cell
def __(
    bench_insert_consume_data,
    bench_insert_mutable_add_data,
    bench_insert_rdf_data,
    bench_insert_sum_data,
):
    bench_insert_data = bench_insert_consume_data + bench_insert_mutable_add_data + bench_insert_sum_data + bench_insert_rdf_data
    return bench_insert_data,


@app.cell
def __(Id, alt, bench_insert_data, experiments, find, mo):
    benchdata = alt.Data(
        values=[
            {
                "label": l.to(str),
                "time/fact (ns)": t.to(int) / c.to(int),
                "#facts": c.to(int),
            }
            for e, l, t, c in find(
                lambda ctx, e, l, t, c: experiments.pattern(ctx, bench_insert_data,
                    [
                        {"experiment": e, "wall_time": t, "element_count": c},
                        {Id: e, "label": l},
                    ],
                )
            )
        ]
    )

    # Create an Altair chart
    benchchart = (
        alt.Chart(benchdata)
        .mark_point()
        .encode(
            x="#facts:Q",  # Encoding along the x-axis
            y="time/fact (ns):Q",  # Encoding along the y-axis
            color="label:O",
        )
    )

    # Make it reactive ⚡
    benchchart = mo.ui.altair_chart(benchchart)
    return benchchart, benchdata


@app.cell
def __(benchchart, mo):
    mo.vstack([benchchart, benchchart.value.head()])
    return


@app.cell
def __(
    bench_query_adhoc_rdf_data,
    bench_query_find_data,
    bench_query_prepared_rdf_data,
):
    bench_query_data = bench_query_find_data + bench_query_adhoc_rdf_data + bench_query_prepared_rdf_data
    return bench_query_data,


@app.cell
def __(Id, alt, bench_query_data, experiments, find, mo):
    benchdata_query = alt.Data(
        values=[
            {"label": l.to(str), "time/fact (ns)": t.to(int) / c.to(int), "#facts": c.to(int)}
            for e, l, t, c in find(
                lambda ctx, e, l, t, c: experiments.pattern(
                    ctx,
                    bench_query_data,
                    [{"experiment": e, "wall_time": t, "element_count": c},
                     {Id: e, "label": l}],
                )
            )
        ]
    )

    # Create an Altair chart
    benchchart_query = (
        alt.Chart(benchdata_query)
        .mark_point()
        .encode(
            x="#facts:Q",  # Encoding along the x-axis
            y="time/fact (ns):Q",  # Encoding along the y-axis
            color="label:O",
        )
    )

    # Make it reactive ⚡
    benchchart_query = mo.ui.altair_chart(benchchart_query)
    return benchchart_query, benchdata_query


@app.cell
def __(benchchart_query, mo):
    mo.vstack([benchchart_query, benchchart_query.value.head()])
    return


@app.cell
def __(Id, alt, bench_query_data, experiments, find, mo):
    benchdata_queryoverhead = alt.Data(
        values=[
            {"label": l.to(str), "time (us)": t.to(int) / 1e3}
            for e, l, t in find(
                lambda ctx, e, l, t: experiments.pattern(
                    ctx,
                    bench_query_data,
                    [{"experiment": e, "wall_time": t, "element_count": 4},
                     {Id: e, "label": l}],
                )
            )
        ]
    )

    # Create an Altair chart
    benchchart_queryoverhead = (
        alt.Chart(benchdata_queryoverhead)
        .mark_bar()
        .encode(
            x="label:O",
            y="time (us):Q",
        )
    )

    # Make it reactive ⚡
    benchchart_queryoverhead = mo.ui.altair_chart(benchchart_queryoverhead)
    return benchchart_queryoverhead, benchdata_queryoverhead


@app.cell
def __(benchchart_queryoverhead, mo):
    mo.vstack([benchchart_queryoverhead, benchchart_queryoverhead.value.head()])
    return


@app.cell
def __(mo):
    mo.md("# Small Worlds")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## dataset
         * distances (nbojects, dim)   f32 matrix    for tests objects
         * neighbors (nbobjects, nbnearest) int32 matrix giving the num of nearest neighbors in train data
          * test      (nbobjects, dim)   f32 matrix  test data
          * train     (nbobjects, dim)   f32 matrix  train data

        load hdf5 data file benchmarks from https://github.com/erikbern/ann-benchmarks
        """
    )
    return


@app.cell
def __():
    mnist_dataset = "/Users/jp/Desktop/triblespace/dorf/datasets/fashion-mnist-784-euclidean.hdf5"
    return mnist_dataset,


@app.cell
def __():
    #dorf.bench_mnist_hnsw(mnist_dataset, True)
    return


@app.cell
def __(bench, mnist_dataset):
    sw = bench.setup(mnist_dataset, 16)
    return sw,


@app.cell
def __(bench, mnist_dataset, sw):
    e = bench.eval(sw, mnist_dataset, 1)
    return e,


@app.cell
def __(e):
    e.avg_cpu_time
    return


@app.cell
def __(e):
    e.avg_distance
    return


if __name__ == "__main__":
    app.run()
