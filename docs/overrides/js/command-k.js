document.addEventListener('keydown', function(e) {
  // Check for Command/Ctrl + K
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault(); // Prevent default browser behavior

    // Find and focus the search input
    const searchInput = document.querySelector('.md-search__input');
    if (searchInput) {
      searchInput.focus();
    }
  }
});
