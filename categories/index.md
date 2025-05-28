---
layout: category
title: "Study shop"
---

## 수리통계학
[수리통계학](/categories/수리통계학/)

### Recent Posts
{% for post in site.posts limit:3 %}
  <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
  <p>{{ post.date | date: "%Y.%m.%d" }} - {{ post.excerpt }}</p>
{% endfor %}