import marimo

__generated_with = "0.6.26"
app = marimo.App()


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
    import maturin_import_hook
    maturin_import_hook.install(settings=maturin_import_hook.MaturinSettings(release=True))
    return maturin_import_hook,


@app.cell
def __():
    import marimo as mo
    return mo,


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
    import dorf
    import importlib
    importlib.reload(dorf)
    from dorf import bench
    from dorf import tribles
    return bench, dorf, importlib, tribles


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
    e.avg_distance
    return


@app.cell
def __(e):
    e.avg_cpu_time
    return


@app.cell
def __():
    import altair as alt
    import vega_datasets
    return alt, vega_datasets


@app.cell
def __(alt, mo, vega_datasets):
    # Load some data
    cars = vega_datasets.data.cars()

    # Create an Altair chart
    chart = alt.Chart(cars).mark_point().encode(
        x='Horsepower', # Encoding along the x-axis
        y='Miles_per_Gallon', # Encoding along the y-axis
        color='Origin', # Category encoding by color
    )

    # Make it reactive ⚡
    chart = mo.ui.altair_chart(chart)
    return cars, chart


@app.cell
def __(chart, mo):
    mo.vstack([chart, chart.value.head()])
    return


@app.cell
def __():
    def attr(x, y):
        return (x, y)
    return attr,


@app.cell
def __(attr):
    hi = attr("hi", "ho")
    return hi,


@app.cell
def __():
    return


@app.cell
def __():
    schemas = {}
    return schemas,


@app.cell
def __(schemas):
    def register_converter(converter):
        global schemas
        k = (converter.schema, converter.type) 
        schemas[k] = converter
        return converter
    return register_converter,


@app.cell
def __(os):
    class Id:
        def __init__(self, bytes):
            self.bytes = bytes
        def gen():
            return Id(os.urandom(16))
        def hex(hex):
            assert len(hex) == 32
            return Id(bytes.fromhex(hex))
    return Id,


@app.cell
def __():
    class ValueSchema:
        @staticmethod
        def pack(value):
            return value
        @staticmethod
        def unpack(bytes):
            return bytes
    return ValueSchema,


@app.cell
def __():
    class RndId:
        entity_id = "DFA138FA94D059161C9AB8C800F6FEC4"
        description = "an random 128 bit id (the first 128bits are zero padding)"
    return RndId,


@app.cell
def __(RndId, register_converter):
    @register_converter
    class RndId_str_Converter:
        schema = RndId
        type = str
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
    @register_converter
    class RndId_Id_Converter:
        schema = RndId
        type = Id
        @staticmethod
        def pack(value):
            return bytes(16) + value.bytes
        @staticmethod
        def unpack(bytes):
            assert all(v == 0 for v in bytes[0: 16])
            assert not all(v == 0 for v in bytes[16: 32])
            return Id(bytes[16:32])
    return RndId_Id_Converter,


@app.cell
def __():
    return


@app.cell
def __():
    class I256BE:
        entity_id = "5F80F30E596C2CEF2AFDDFCBD9933AC7"
        description = "an signed 256bit integer in big endian encoding"
    return I256BE,


@app.cell
def __():
    class I256LE:
        entity_id = "F5E93737BFD910EDE8902ACAA8493CEE"
        description = "a signed 256bit integer in little endian encoding"
    return I256LE,


@app.cell
def __():
    class U256BE:
        entity_id = "5E868BA4B9C06DD12E7F4AA064D1A7C7"
        description = "an unsigned 256bit integer in big endian encoding"
    return U256BE,


@app.cell
def __():
    class U256LE:
        entity_id = "EC9C2F8C3C3156BD203D92888D7479CD"
        description = "an unsigned 256bit integer in little endian encoding"
    return U256LE,


@app.cell
def __(I256BE, register_converter):
    @register_converter
    class I256BE_Int_Converter:
        schema = I256BE
        type = int
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=True)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=True)
    return I256BE_Int_Converter,


@app.cell
def __(I256LE, register_converter):
    @register_converter
    class I256LE_Int_Converter:
        schema = I256LE
        type = int
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='little', signed=True)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='little', signed=True)
    return I256LE_Int_Converter,


@app.cell
def __(U256BE, register_converter):
    @register_converter
    class U256BE_Int_Converter:
        schema = U256BE
        type = int
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=False)
    return U256BE_Int_Converter,


@app.cell
def __(U256LE, register_converter):
    @register_converter
    class U256LE_Int_Converter:
        schema = U256LE
        type = int
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='little', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='little', signed=False)
    return U256LE_Int_Converter,


@app.cell
def __(schemas):
    class Value:
        def __init__(self, schema, bytes):
            self.schema = schema
            self.bytes = bytes

        @staticmethod
        def of(schema, value):
            global schemas
            t = type(value)
            b = schemas[(schema, t)].pack(value)
            return Value(schema, b)

        def to(self, type):
            global schemas
            return schemas[(self.schema, type)].unpack(self.bytes)
    return Value,


@app.cell
def __():
    class NSDuration:
        entity_id = "BD1DA74AABF1D01A5CF4EEF3683B1EC5"
        description = "a time duration in nanoseconds stored as a signed 256bit big endian integer"
    return NSDuration,


@app.cell
def __(NSDuration, register_converter):
    @register_converter
    class NSDuration_Int_Converter:
        schema = NSDuration
        type = int
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=False)
    return NSDuration_Int_Converter,


@app.cell
def __():
    class FR256LE:
        entity_id = "77694E74654A039625FA5911381F3897"
        description = "a unitless fraction stored as a (numerator, denominator) pair of signed 128bit little endian integers"
    return FR256LE,


@app.cell
def __(FR256LE, fractions, register_converter):
    @register_converter
    class FR128LE_Fraction_Converter:
        schema = FR256LE
        type = fractions.Fraction
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
def __(FR256LE, Value, fractions):
    Value.of(FR256LE, fractions.Fraction(-123, 314)).to(fractions.Fraction)
    return


@app.cell
def __(U256LE, Value):
    Value(U256LE, bytes(32)).to(int)
    return


@app.cell
def __(U256LE, Value):
    Value.of(U256LE, 1).to(int)
    return


@app.cell
def __():
    import time
    return time,


@app.cell
def __(time):
    type(time.time_ns())
    return


@app.cell
def __(Id, Value, tribles):
    class Namespace:
        def __init__(self, declaration):
            self.declaration = declaration

        def entity(self, entity):
            if Id in entity:
                id = entity[Id]
                eb = id.bytes
                assert len(eb) == 16
            else:
                eb = Id.gen().bytes;

            tribledata = bytearray()
            for key, value in entity.items():
                attr_id = self.declaration[key][1];
                attr_schema = self.declaration[key][0];
                value = Value.of(attr_schema, value);

                ab = attr_id.bytes
                assert len(eb) == 16

                vb = value.bytes
                assert len(vb) == 32

                tribledata.extend(eb)
                tribledata.extend(ab)
                tribledata.extend(vb)

            return tribles.PyTribleSet.from_bytes(bytes(tribledata))
        
        def pattern(self, ctx, set, entities):
            return []
    return Namespace,


@app.cell
def __(Namespace):
    def ns(declaration):
        return Namespace(declaration)
    return ns,


@app.cell
def __(FR256LE, Id, NSDuration, RndId, U256LE, ns):
    experiments = ns({
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
def __():
    import fractions
    from fractions import Fraction
    return Fraction, fractions


@app.cell
def __():
    import timeit
    return timeit,


@app.cell
def __():
    element_count_exp = 4
    return element_count_exp,


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
def __(gen_data, tribles):
    def bench_consume(size):
        set = tribles.PyTribleSet.empty()
        for entity in gen_data(size):
            set.consume(entity)
        return set
    return bench_consume,


@app.cell
def __(gen_data, tribles):
    def bench_mutable_add(size):
        set = tribles.PyTribleSet.empty()
        for entity in gen_data(size):
            set += entity
        return set
    return bench_mutable_add,


@app.cell
def __(gen_data, tribles):
    def bench_sum(size):
        set = sum(gen_data(size), start = tribles.PyTribleSet.empty())
        return set
    return bench_sum,


@app.cell
def __(timeit):
    def time_ns(l):
        s = timeit.timeit(lambda: l, number=1)
        return int(s * 1e9)
    return time_ns,


@app.cell
def __(
    Id,
    bench_consume,
    element_count_exp,
    experiments,
    time_ns,
    tribles,
):
    _experiment = Id.gen()
    bench_consume_data = sum([experiments.entity({
        "experiment": _experiment,
        "wall_time": time_ns(lambda: bench_consume(2 ** i)),
        "element_count": (2 ** i) * 4 }) for i in range(element_count_exp)], tribles.PyTribleSet.empty())
    return bench_consume_data,


@app.cell
def __(
    Id,
    bench_mutable_add,
    element_count_exp,
    experiments,
    time_ns,
    tribles,
):
    _experiment = Id.gen()
    bench_mutable_add_data = sum([experiments.entity({
        "experiment": _experiment,
        "wall_time": time_ns(lambda: bench_mutable_add(2 ** i)),
        "element_count": (2 ** i) * 4 }) for i in range(element_count_exp)], tribles.PyTribleSet.empty())
    return bench_mutable_add_data,


@app.cell
def __(Id, bench_sum, element_count_exp, experiments, time_ns, tribles):
    _experiment = Id.gen()
    bench_sum_data = sum([experiments.entity({
        "experiment": _experiment,
        "wall_time": time_ns(lambda: bench_sum(2 ** i)),
        "element_count": (2 ** i) * 4 }) for i in range(element_count_exp)], tribles.PyTribleSet.empty())
    return bench_sum_data,


@app.cell
def __(mo):
    mo.md("# RDFLib")
    return


@app.cell
def __():
    from rdflib import Graph, URIRef, BNode, Literal, Namespace as RDFNamespace
    return BNode, Graph, Literal, RDFNamespace, URIRef


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
def __(Id, bench_rdf, element_count_exp, experiments, time_ns, tribles):
    _experiment = Id.gen()
    bench_rdf_data = sum([experiments.entity({
        "experiment": _experiment,
        "wall_time": time_ns(lambda: bench_rdf(2 ** i)),
        "element_count": (2 ** i) * 4 }) for i in range(element_count_exp)], tribles.PyTribleSet.empty())
    return bench_rdf_data,


@app.cell
def __(
    bench_consume_data,
    bench_mutable_add_data,
    bench_rdf_data,
    bench_sum_data,
):
    bench_combined_data = bench_consume_data + bench_mutable_add_data + bench_sum_data + bench_rdf_data
    return bench_combined_data,


@app.cell
def __(alt, bench_combined_data, experiments, find, mo):
    benchdata = alt.Data(values=list(find(
        lambda ctx, e, t, c:
            experiments.pattern(ctx, bench_combined_data, [{
                "experiment": e,
                "wall_time": t,
                "element_count": c}]))))

    # Create an Altair chart
    benchchart = alt.Chart(benchdata).mark_point().encode(
        x='c:Q', # Encoding along the x-axis
        y='t:Q', # Encoding along the y-axis
        color='e:O'
    )

    # Make it reactive ⚡
    benchchart = mo.ui.altair_chart(benchchart)
    return benchchart, benchdata


@app.cell
def __(benchchart, mo):
    mo.vstack([benchchart, benchchart.value.head()])
    return


@app.cell
def __():
    class Query:
        def __init__(self, constraint):
            self.constraint = constraint

        def run(self):
            for c in self.constraint:
                yield c

    return Query,


@app.cell
def __(Query):
    def find(query):
        variable_names = query.__code__.co_varnames[1:]
        constraint = query(None, *variable_names)
        execution = Query(constraint)
        for result in execution.run():
            yield result
    return find,


@app.cell
def __(bench_combined_data, experiments, find):
    find(lambda ctx, experiment, time, count:
        experiments.pattern(ctx, bench_combined_data, [{
            "experiment": experiment,
            "wall_time": time,
            "element_count": count}]))
    return


@app.cell
def __(bench_combined_data, experiments, find):
    list(find(lambda ctx, experiment, time, count:
        experiments.pattern(ctx, bench_combined_data, [{
            "experiment": experiment,
            "wall_time": time,
            "element_count": count}])))
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
