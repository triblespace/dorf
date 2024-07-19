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
def __(mo):
    import altair as alt
    import vega_datasets

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
    return alt, cars, chart, vega_datasets


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
def __(F256BE, U256BE, tribles):
    experiments = {
        "avg_cpu_time": (tribles.types.Duration, "1C333940F98D0CFCEBFCC408FA35FF92"),
        "avg_distance": (F256BE, "78D9B9230C044FA4E1585AFD14CFB3EE"),
        "avg_cpu_time": (tribles.types.Duration, "999BF50FFECF9C0B62FD23689A6CA0D0"),
        "change_count": (U256BE, "AD5DD3F72FA8DD67AF0D0DA5298A98B9"),
        "layer_explored": (U256BE, "2DB0F43553543173C42C8AE1573A38DB"),
    }
    return experiments,


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
    class ValueSchema:
        @staticmethod
        def pack(value):
            value
        @staticmethod
        def unpack(bytes):
            bytes
    return ValueSchema,


@app.cell
def __():
    class I256BE:
        description = "an signed 256bit integer in big endian encoding"
    return I256BE,


@app.cell
def __():
    class I256LE:
        description = "a signed 256bit integer in little endian encoding"
    return I256LE,


@app.cell
def __():
    class U256BE:
        description = "an unsigned 256bit integer in big endian encoding"
    return U256BE,


@app.cell
def __():
    class U256LE:
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
            value.to_bytes(32, byteorder='big', signed=True)
        @staticmethod
        def unpack(bytes):
            int.from_bytes(bytes, byteorder='big', signed=True)
    return I256BE_Int_Converter,


@app.cell
def __(I256LE, register_converter):
    @register_converter
    class I256LE_Int_Converter:
        schema = I256LE
        type = int
        @staticmethod
        def pack(value):
            value.to_bytes(32, byteorder='little', signed=True)
        @staticmethod
        def unpack(bytes):
            int.from_bytes(bytes, byteorder='little', signed=True)
    return I256LE_Int_Converter,


@app.cell
def __(U256BE, register_converter):
    @register_converter
    class U256BE_Int_Converter:
        schema = U256BE
        type = int
        @staticmethod
        def pack(value):
            value.to_bytes(32, byteorder='big', signed=False)
        @staticmethod
        def unpack(bytes):
            int.from_bytes(bytes, byteorder='big', signed=False)
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
        description = "a time duration in nanoseconds stored as a signed 256bit big endian integer"
    return NSDuration,


@app.cell
def __(Duration, NSDuration, register_converter):
    @register_converter
    class NSDuration_Duration_Converter:
        schema = NSDuration
        type = Duration
        @staticmethod
        def pack(value):
            return value.to_bytes(32, byteorder='big', signed=False)
        @staticmethod
        def unpack(bytes):
            return int.from_bytes(bytes, byteorder='big', signed=False)
    return NSDuration_Duration_Converter,


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
def __():
    return


if __name__ == "__main__":
    app.run()
