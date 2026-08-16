const filterButtons = document.querySelectorAll(".archive-filter");
const archivePosts = document.querySelectorAll(".archive-post");
const emptyMessage = document.querySelector(".archive-empty");

filterButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const selected = button.dataset.filter;
    let visibleCount = 0;

    filterButtons.forEach((item) => {
      const isActive = item === button;
      item.classList.toggle("is-active", isActive);
      item.setAttribute("aria-pressed", String(isActive));
    });

    archivePosts.forEach((post) => {
      const isVisible = selected === "all" || post.dataset.category === selected;
      post.hidden = !isVisible;
      if (isVisible) visibleCount += 1;
    });

    if (emptyMessage) emptyMessage.hidden = visibleCount !== 0;
  });
});
