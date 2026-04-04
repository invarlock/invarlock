FROM gcr.io/oss-fuzz-base/base-builder-python@sha256:d78540f9e04918ea3c903f5280c597aa48643a6e01d2b86c38f4101807433212

COPY . $SRC/invarlock
WORKDIR $SRC/invarlock
COPY ./.clusterfuzzlite/build.sh $SRC/
