from invoke import task


@task
def clean(ctx, pycache=True):
    """ Remove all build artifacts. """
    if pycache:
        ctx.run("find . -type f -name '*.pyc' -delete")
    ctx.run("rm -rf build dist")


@task
def build(ctx):
    """ Build cuda module. """
    ctx.run("cd extensions")
    ctx.run("python setup.py install")
