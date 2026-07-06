default: build-submodule

.PHONY:
update-submodule:
	# create a clean (maybe updated) copy of whisper.cpp
	rsync ../../ggml.c              Sources/whisper/
	rsync ../../ggml-alloc.c        Sources/whisper/
	rsync ../../ggml-alloc.h        Sources/whisper/
	rsync ../../ggml-backend.c      Sources/whisper/
	rsync ../../ggml-backend.h      Sources/whisper/
	rsync ../../ggml-backend-impl.h Sources/whisper/
	rsync ../../ggml-common.h       Sources/whisper/
	rsync ../../ggml-impl.h         Sources/whisper/
	rsync ../../ggml-metal.h        Sources/whisper/
	rsync ../../ggml-metal.m        Sources/whisper/
	rsync ../../ggml-metal.metal    Sources/whisper/
	rsync ../../ggml-quants.c       Sources/whisper/
	rsync ../../ggml-quants.h       Sources/whisper/
	rsync ../../whisper.cpp         Sources/whisper/
	rsync ../../ggml.h              Sources/whisper/include/
	rsync ../../whisper.h           Sources/whisper/include/
	rsync ../../coreml/*            Sources/whisper/coreml/

SOURCES := $(shell find Sources/ -print)
.build: $(SOURCES)
	swift build

.PHONY:
build-submodule: update-submodule Package.swift .build
	touch publish-trigger

.PHONY:
build: Package.swift .build

.PHONY:
test: build
	./.build/debug/test-objc
	./.build/debug/test-swift

.PHONY:
publish: publish-trigger
	@echo " \
		\n\
		cd /path/to/whisper.cpp/bindings/ios\n\
		git commit\n\
		git tag 1.6.2\n\
		git push origin master --tags\n\
		"

clean:
	rm -rf .build
