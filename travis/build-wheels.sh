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

mkdir /io/good_wheelhouse
cp /io/wheelhouse/*dwave* /io/good_wheelhouse/

# Install packages and test
for PYBIN in /opt/python/*/bin/; do
    if [[ ${PYBIN} =~ 26|33 ]]; then
        # numpy doesn't support 2.6 or 3.3, just skip em
        continue
    fi
    "${PYBIN}/pip" install dwave_sage --no-index -f /io/good_wheelhouse/
    (cd "/io/dw_sa_chi"; "${PYBIN}/python" -m unittest discover)
done
