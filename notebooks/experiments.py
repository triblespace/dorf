import marimo

__generated_with = "0.6.23"
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
    import smol
    import importlib
    importlib.reload(smol)
    return importlib, smol


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
    mnist_dataset = "/Users/jp/Desktop/triblespace/smol/datasets/fashion-mnist-784-euclidean.hdf5"
    return mnist_dataset,


@app.cell
def __(mnist_dataset, smol):
    smol.bench_mnist(mnist_dataset, True)
    return


@app.cell
def __(mo, smol):
    with mo.redirect_stdout():
        smol.printstuff()
    return


@app.cell
def __(smol):
    smol.printstuff()
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
