#!/bin/bash
set -e -x

for PYBIN in /opt/python/*/bin; do
    if [[ ${PYBIN} =~ 26|33 ]]; then
        # numpy doesn't support 2.6 or 3.3, just skip em
        continue
    fi
    "${PYBIN}/pip" install -r /io/requirements.txt
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
    # install coverage
    "${PYBIN}/pip" install coverage
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
    "${PYBIN}/pip" install dwave_sage --no-index -f /io/wheelhouse/
    # -a option on coverage run just appends to the same file so it doesn't
    # get overwritten
    (cd /io/; ls -al; "${PYBIN}/coverage" run --source=dwave_sage -a -m unittest discover)
    (cd /io/; "${PYBIN}/python" -m unittest discover)
done
