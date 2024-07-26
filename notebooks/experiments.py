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
def __():
    class Id:
        def __init__(self, bytes):
            self.bytes = bytes
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
def __(Id):
    def id(hex):
        assert len(hex) == 32
        return Id(bytes.fromhex(hex))
    return id,


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
def __(Value, os, tribles):
    class Namespace:
        def __init__(self, declaration):
            self.declaration = declaration
            
        def entity(self, entity, id = None):
            if id is not None:
                eb = id.bytes
                assert len(eb) == 16
            else:
                eb = os.urandom(16);
            
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

    return Namespace,


@app.cell
def __(Namespace):
    def ns(declaration):
        return Namespace(declaration)
    return ns,


@app.cell
def __(FR256LE, NSDuration, U256BE, id, ns):
    experiments = ns({
        "avg_cpu_time": (NSDuration, id("1C333940F98D0CFCEBFCC408FA35FF92")),
        "avg_distance": (FR256LE, id("78D9B9230C044FA4E1585AFD14CFB3EE")),
        "avg_wall_time": (NSDuration, id("999BF50FFECF9C0B62FD23689A6CA0D0")),
        "change_count": (U256BE, id("AD5DD3F72FA8DD67AF0D0DA5298A98B9")),
        "layer_explored": (U256BE, id("2DB0F43553543173C42C8AE1573A38DB")),
    })
    return experiments,


@app.cell
def __():
    import fractions
    from fractions import Fraction
    return Fraction, fractions


@app.cell
def __(Fraction, experiments, id):
    a = experiments.entity({
        "layer_explored": 0,
        "avg_cpu_time": 1000,
        "avg_distance": Fraction(13, 17)
    }, id("40604EA0874193604EC87F252125DFE3"))
    return a,


@app.cell
def __(Fraction, experiments):
    b = experiments.entity({
        "layer_explored": 1,
        "avg_cpu_time": 500,
        "avg_distance": Fraction(15, 23)
    })
    return b,


@app.cell
def __(a, b):
    a + b
    return


@app.cell
def __():
    import timeit
    return timeit,


@app.cell
def __():
    return


@app.cell
def __(Fraction, experiments):
    def gen_data(size):
        for i in range(size):
            yield experiments.entity({
                "layer_explored": i,
                "avg_cpu_time": 500 * i,
                "avg_wall_time": 600 * i,
                "avg_distance": Fraction(i, 1)})
    return gen_data,


@app.cell
def __():
    return


@app.cell
def __(gen_data, timeit, tribles):
    timeit.timeit(lambda: sum(gen_data(1000), start = tribles.PyTribleSet.empty()), number=1)
    return


@app.cell
def __(alt, gen_data, mo, timeit, tribles):
    benchdata = alt.Data(values=[{"t": timeit.timeit(lambda: sum(gen_data(2 ** i), start = tribles.PyTribleSet.empty()), number=1),
             "n": (2 ** i) * 4 } for i in range(21)])

    # Create an Altair chart
    benchchart = alt.Chart(benchdata).mark_point().encode(
        x='n:Q', # Encoding along the x-axis
        y='t:Q', # Encoding along the y-axis
    )

    # Make it reactive ⚡
    benchchart = mo.ui.altair_chart(benchchart)
    return benchchart, benchdata


@app.cell
def __(benchchart, mo):
    mo.vstack([benchchart, benchchart.value.head()])
    return


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
def __():
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
def __(alt, bench_rdf, mo, timeit):
    rdfbenchdata = alt.Data(values=[{"t": timeit.timeit(lambda: bench_rdf(2 ** i), number=1),
             "n": (2 ** i) * 4 } for i in range(21)])

    # Create an Altair chart
    rdfbenchchart = alt.Chart(rdfbenchdata).mark_point().encode(
        x='n:Q', # Encoding along the x-axis
        y='t:Q', # Encoding along the y-axis
    )

    # Make it reactive ⚡
    rdfbenchchart = mo.ui.altair_chart(rdfbenchchart)
    return rdfbenchchart, rdfbenchdata


@app.cell
def __(mo, rdfbenchchart):
    mo.vstack([rdfbenchchart, rdfbenchchart.value.head()])
    return


app._unparsable_cell(
    r"""
    def bench_consume(size):
        let set = tribles.PyTribleSet.empty()
        for i in range(size):
            set.consume(experiments.entity({
                \"layer_explored\": i,
                \"avg_cpu_time\": 500 * i,
                \"avg_wall_time\": 600 * i,
                \"avg_distance\": Fraction(i, 1)}))
        return set
    """,
    name="__"
)


@app.cell
def __(alt, bench_consume, mo, timeit):
    consumebenchdata = alt.Data(values=[{"t": timeit.timeit(lambda: bench_consume(2 ** i), number=1),
             "n": (2 ** i) * 4 } for i in range(21)])

    # Create an Altair chart
    consumebenchchart = alt.Chart(consumebenchdata).mark_point().encode(
        x='n:Q', # Encoding along the x-axis
        y='t:Q', # Encoding along the y-axis
    )

    # Make it reactive ⚡
    consumebenchchart = mo.ui.altair_chart(consumebenchchart)
    return consumebenchchart, consumebenchdata


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
