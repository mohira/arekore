.PHONY: test
test:
	poetry run mypy .
	poetry run pytest
	poetry run pytest --cov-report term-missing --cov=arekore/ tests/

.PHONY: testpypi
testpypi:
	# poetry config http-basic.testpypi {{USER_NAME}} {{PASSWORD}}
	poetry lock
	poetry build
	poetry publish -r testpypi

.PHONY: publish
publish:
	poetry publish --build
