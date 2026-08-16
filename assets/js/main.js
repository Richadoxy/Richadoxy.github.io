(function () {
  const root = document.documentElement;
  const languageButton = document.querySelector(".language-toggle");
  const navButton = document.querySelector(".nav-toggle");
  const nav = document.querySelector(".site-nav");

  function setLanguage(language) {
    const nextLanguage = language === "zh" ? "zh" : "en";
    root.dataset.language = nextLanguage;
    root.lang = nextLanguage === "zh" ? "zh-CN" : "en";
    try {
      localStorage.setItem("xiyue-site-language", nextLanguage);
    } catch (_error) {
      // The site still works when storage is blocked.
    }
  }

  try {
    const savedLanguage = localStorage.getItem("xiyue-site-language");
    if (savedLanguage) setLanguage(savedLanguage);
  } catch (_error) {
    // Keep English as the default language.
  }

  languageButton?.addEventListener("click", function () {
    setLanguage(root.dataset.language === "en" ? "zh" : "en");
  });

  navButton?.addEventListener("click", function () {
    const isOpen = nav.classList.toggle("is-open");
    navButton.setAttribute("aria-expanded", String(isOpen));
  });

  nav?.querySelectorAll("a").forEach(function (link) {
    link.addEventListener("click", function () {
      nav.classList.remove("is-open");
      navButton?.setAttribute("aria-expanded", "false");
    });
  });
})();
