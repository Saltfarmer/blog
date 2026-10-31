---
title: "Start here"
layout: single
permalink: /start-here/
author_profile: false
toc: false
comments: false
share: false
related: false
excerpt: "New here? Follow a series in order, or pick a topic."
---

New here? This blog is a mix of data science tutorials, notes from learning new tools, and personal journal posts. Pick a path below.

{% assign featured = site.posts | where: "featured", true %}
{% if featured.size > 0 %}
## Best of the blog

<ul>
{% for post in featured %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title | strip_html }}</a></li>
{% endfor %}
</ul>
{% endif %}

## Follow a series in order

{% assign series_posts = site.posts | where_exp: "p", "p.series" | group_by: "series" | sort: "name" %}
{% for group in series_posts %}
{% assign ordered = group.items | sort: "series_order" %}
### {{ group.name }}

<p class="muted-count">{{ ordered.size }} parts</p>

<ol>
{% for post in ordered %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title | strip_html }}</a></li>
{% endfor %}
</ol>
{% endfor %}

## Browse by topic

{% assign topics = site.categories | sort %}
<ul>
{% for topic in topics %}
  <li><a href="{{ '/categories/' | relative_url }}#{{ topic[0] | slugify }}">{{ topic[0] }}</a> ({{ topic[1].size }})</li>
{% endfor %}
</ul>

## Prefer to search?

<a class="btn btn--primary" href="{{ '/search/' | relative_url }}">Search the blog</a>
