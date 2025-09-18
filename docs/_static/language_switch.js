// Language switching functionality for RecIS documentation

document.addEventListener('DOMContentLoaded', function() {
    // Create language switcher
    createLanguageSwitcher();
    
    // Add language indicators to links
    addLanguageIndicators();
});

function createLanguageSwitcher() {
    const switcher = document.createElement('div');
    switcher.className = 'language-switcher';
    
    const select = document.createElement('select');
    select.innerHTML = `
        <option value="zh">中文</option>
        <option value="en">English</option>
    `;
    
    // Detect current language from URL
    const currentLang = detectCurrentLanguage();
    select.value = currentLang;
    
    select.addEventListener('change', function() {
        switchLanguage(this.value);
    });
    
    switcher.appendChild(select);
    
    // Insert into header
    const header = document.querySelector('.wy-nav-top');
    if (header) {
        header.appendChild(switcher);
    }
}

function detectCurrentLanguage() {
    const path = window.location.pathname;
    if (path.includes('_en.html') || path.includes('/en/')) {
        return 'en';
    }
    return 'zh';
}

function switchLanguage(targetLang) {
    const currentPath = window.location.pathname;
    let newPath;
    
    if (targetLang === 'en') {
        // Switch to English
        if (currentPath.includes('index.html')) {
            newPath = currentPath.replace('index.html', 'index_en.html');
        } else if (currentPath.includes('.html')) {
            newPath = currentPath.replace('.html', '_en.html');
        } else {
            newPath = currentPath + '_en.html';
        }
    } else {
        // Switch to Chinese
        if (currentPath.includes('_en.html')) {
            newPath = currentPath.replace('_en.html', '.html');
        } else if (currentPath.includes('index_en.html')) {
            newPath = currentPath.replace('index_en.html', 'index.html');
        } else {
            newPath = currentPath;
        }
    }
    
    // Check if target page exists, otherwise go to main index
    fetch(newPath, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                window.location.href = newPath;
            } else {
                // Fallback to main index
                const fallbackPath = targetLang === 'en' ? '/index_en.html' : '/index.html';
                window.location.href = fallbackPath;
            }
        })
        .catch(() => {
            // Fallback to main index
            const fallbackPath = targetLang === 'en' ? '/index_en.html' : '/index.html';
            window.location.href = fallbackPath;
        });
}

function addLanguageIndicators() {
    // Add language indicators to navigation links
    const links = document.querySelectorAll('.wy-menu-vertical a');
    links.forEach(link => {
        const href = link.getAttribute('href');
        if (href && href.includes('_en.html')) {
            link.classList.add('lang-en');
        } else if (href && href.includes('.html')) {
            link.classList.add('lang-zh');
        }
    });
}

// Add keyboard shortcut for language switching
document.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.shiftKey && e.key === 'L') {
        const currentLang = detectCurrentLanguage();
        const targetLang = currentLang === 'zh' ? 'en' : 'zh';
        switchLanguage(targetLang);
    }
});