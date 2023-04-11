python ?= python3

all: test-dist

dist:
	@$(MAKE) --no-print-directory clean-dist
	$(python) -m build
	@$(MAKE) --no-print-directory clean-egg-info

test:
	@$(python) -c "import os; import cycept; \
print('Testing', os.path.dirname(cycept.__file__)); cycept.test()"
.PHONY: test

test-dist:
	@$(MAKE) --no-print-directory clean-dist clean-venv
	$(python) -m venv venv
	. ./venv/bin/activate && ./venv/bin/python -m pip install --upgrade pip
	. ./venv/bin/activate && ./venv/bin/python -m pip install build
	@. ./venv/bin/activate && python=./venv/bin/python $(MAKE) \
--no-print-directory dist
	@$(MAKE) --no-print-directory clean-tmp
	mkdir -p tmp
	cd tmp && . ../venv/bin/activate && ../venv/bin/python -m pip install \
$$(echo ../dist/*.whl)[test]
	cd tmp && . ../venv/bin/activate && python=../venv/bin/python $(MAKE) \
--no-print-directory -f ../Makefile test
	@$(MAKE) --no-print-directory clean-tmp
.PHONY: test-dist

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

