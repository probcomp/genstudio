

fswatch -o intro_slides.py | while read; do
  echo "Change detected in intro_slides.py. Syncing and converting to slides..."
  jupytext --sync intro_slides.py && jupyter nbconvert --to slides --execute --ExecutePreprocessor.kernel_name=studio intro_slides.ipynb
  echo "Sync and convert complete."
done

# jupyter nbconvert --to slides --execute --ExecutePreprocessor.kernel_name=studio intro_slides.ipynb
