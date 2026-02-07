.PHONY: boot audit migrate clean

boot:
	@echo "⚡ Initializing Level 10 Sequence..."
	@python3 boot.py

audit:
	@echo "🔍 Scanning Project Structure..."
	@# Excludes the noise folders for a clean view
	@tree -I 'wandb|__pycache__|Google Gemini_files|graveyard'

migrate:
	@echo "🚀 Executing Golden Schema Migration..."
	@python3 tools/golden_schema_executor.py

clean:
	@echo "🧹 Cleaning compiled python files..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
