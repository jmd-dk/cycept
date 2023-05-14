python ?= python3

all: test-dist

dist: cycept cycept/tests pyproject.toml README.md CHANGELOG.md LICENSE
	@$(MAKE) --no-print-directory clean-dist
	$(python) -m build
	@$(MAKE) --no-print-directory clean-egg-info

test-dist:
	@$(MAKE) --no-print-directory clean-dist clean-venv clean-tmp
	$(python) -m venv venv
	. ./venv/bin/activate && ./venv/bin/python -m pip install --upgrade pip
	. ./venv/bin/activate && ./venv/bin/python -m pip install build
	@. ./venv/bin/activate && python=./venv/bin/python $(MAKE) \
--no-print-directory dist
	mkdir -p tmp
	cd tmp && . ../venv/bin/activate && ../venv/bin/python -m pip install \
$$(echo ../dist/*.whl)[repl,test,bench]
	@cd tmp && . ../venv/bin/activate && python=../venv/bin/python $(MAKE) \
--no-print-directory -f ../Makefile test-all
	@$(MAKE) --no-print-directory clean-tmp
.PHONY: test-dist

define test
    $(python) -c "import pathlib; import cycept; \
print('Testing from', pathlib.Path(cycept.__file__).parent); cycept.test($(1))"
endef

test-all:
	@$(call test)
.PHONY: test-all

test-cycept:
	@$(call test,'cycept')
.PHONY: test-cycept

test-bench:
	@$(call test,'bench')
.PHONY: test-bench

bench:
	@$(python) -c "import pathlib; import cycept; \
print('Benchmarking from', pathlib.Path(cycept.__file__).parent); cycept.bench()"
.PHONY: test

clean-dist:
	$(RM) -r dist

clean-egg-info:
	$(RM) -r *.egg-info

clean-tmp:
	$(RM) -r tmp

clean-venv:
	$(RM) -r venv

clean: clean-dist clean-tmp clean-egg-info clean-venv
	$(RM) -r __pycache__ */__pycache__ */*/__pycache__ .pytest_cache

.PHONY:            \
    clean-dist     \
    clean-egg-info \
    clean-tmp      \
    clean-venv     \
    clean          \

