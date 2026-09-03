# Xiyue Dong — Personal Homepage

A lightweight bilingual academic homepage inspired by the information architecture of Academic Pages. It is plain HTML, CSS, and JavaScript, so it can be hosted directly on GitHub Pages without a build step.

## Preview locally

From this directory, run:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

## Information intentionally left blank

- Google Scholar profile
- CV file
- Paper, code, video, and project-page links
- Project demos and research notes

Search for `placeholder`, `pending`, `TBD`, or `待补充` in `index.html` to find the main unfinished fields.

## Deploy to GitHub Pages

1. Create a repository named `Richadoxy.github.io`.
2. Copy the contents of this directory to the repository root.
3. Push to the default branch.
4. In GitHub repository settings, enable Pages deployment from the default branch and root directory.

No Ruby, Jekyll, Node.js, or database is required.

## Rebuild the Learning Journey blog

The generated blog lives in `blog/`. Its Markdown sources and Draw.io files are copied from the BrainCo-IL documentation directory so the published site remains self-contained.

```bash
python3 -m venv .blog-build-env
.blog-build-env/bin/pip install Markdown==3.8.2
.blog-build-env/bin/python scripts/build_blog.py
```

The script rebuilds the learning-path article pages from the bundled Markdown in `blog/source/`. The bundled files are the canonical website sources, so a normal build never overwrites local edits.

To explicitly refresh matching engineering notes and Draw.io diagrams from the external BrainCo documentation directory, opt in with:

```bash
.blog-build-env/bin/python scripts/build_blog.py --sync-upstream
```

The archive is organized into six dependency-first series: foundations, imitation learning, VLA architectures, engineering, runtime/deployment, and tactile VLA. Article codes such as `F01`, `IL01`, and `VLA00` express the reading path without changing the stable page URLs.
