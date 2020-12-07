.PHONY: build
build: 
	poetry run python encode.py
	cat .build/script.py > encoded
.PHONY: archive
archive:
	git archive HEAD -o submission.zip
	zip -ur submission.zip working/cache/*

