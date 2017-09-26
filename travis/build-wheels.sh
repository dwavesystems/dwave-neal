#!/bin/bash
set -e -x

for PYBIN in /opt/python/*/bin; do
    if [[ ${PYBIN} =~ 26|33 ]]; then
        # numpy doesn't support 2.6 or 3.3, just skip em
        continue
    fi
    "${PYBIN}/pip" install -r /io/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    if ! [[ "$whl" =~ dwave_sage ]]; then
        continue
    fi
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ ${PYBIN} =~ 26|33 ]]; then
        # numpy doesn't support 2.6 or 3.3, just skip em
        continue
    fi
    "${PYBIN}/pip" install dwave_sage --no-index -f /io/wheelhouse
    (cd "$HOME"; "${PYBIN}/python" /io/dw_sa_chi/tests/test_python_sa_wrapper.py)
done
