const JSON_HEADERS = { 'Content-Type': 'application/json' };

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...JSON_HEADERS,
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    return response.json();
  }
  return response.text();
}

function buildQuery(filters, category, sort, order, limit, page) {
  const params = new URLSearchParams();
  params.set('category', category);
  params.set('sort', sort);
  params.set('order', order);
  params.set('limit', String(limit));
  params.set('offset', String(page * limit));
  if (filters.query) {
    params.set('q', filters.query);
  }
  filters.languages.forEach((value) => params.append('language', value));
  filters.statuses.forEach((value) => params.append('status', value));
  if (filters.minSubscribers != null && filters.minSubscribers !== '') {
    params.set('min_subscribers', String(filters.minSubscribers));
  }
  if (filters.maxSubscribers != null && filters.maxSubscribers !== '') {
    params.set('max_subscribers', String(filters.maxSubscribers));
  }
  params.set('emails_only', filters.emailsOnly ? 'true' : 'false');
  params.set('unique_emails', filters.uniqueEmails ? 'true' : 'false');
  params.set('email_gate_only', filters.emailGateOnly ? 'true' : 'false');
  return params.toString();
}

export async function fetchChannels(category, filters, sort, order, limit, page) {
  const query = buildQuery(filters, category, sort, order, limit, page);
  return request(`/api/channels?${query}`);
}

export async function archiveChannels(channelIds) {
  return request('/api/channels/archive_bulk', {
    method: 'POST',
    body: JSON.stringify({ channel_ids: channelIds }),
  });
}

export async function archiveByFilter(category, filters, sort, order, limit, page) {
  const query = buildQuery(filters, category, sort, order, limit, page);
  const body = {};
  if (filters.emailsOnly) {
    body.filter = 'emails_only';
  }
  return request(`/api/channels/archive_bulk?${query}`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function blacklistChannels(channelIds, category) {
  const query = category ? `?category=${encodeURIComponent(category)}` : '';
  return request(`/api/channels/blacklist_bulk${query}`, {
    method: 'POST',
    body: JSON.stringify({ channel_ids: channelIds }),
  });
}

export async function blacklistByFilter(category, filters, sort, order, limit, page) {
  const query = buildQuery(filters, category, sort, order, limit, page);
  const body = {};
  if (filters.emailsOnly) {
    body.filter = 'emails_only';
  }
  return request(`/api/channels/blacklist_bulk?${query}`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function restoreChannels(channelIds, category) {
  const query = `?category=${encodeURIComponent(category)}`;
  return request(`/api/channels/restore_bulk${query}`, {
    method: 'POST',
    body: JSON.stringify({ channel_ids: channelIds }),
  });
}

export async function restoreByFilter(category, filters, sort, order, limit, page) {
  const query = buildQuery(filters, category, sort, order, limit, page);
  const body = {};
  if (filters.emailsOnly) {
    body.filter = 'emails_only';
  }
  return request(`/api/channels/restore_bulk?${query}`, {
    method: 'POST',
    body: JSON.stringify(body),
  });
}

export async function discoverChannels(keywords, perKeyword, options = {}) {
  const payload = { keywords, perKeyword };
  if (options.lastUploadMaxAgeDays != null) {
    payload.last_upload_max_age_days = options.lastUploadMaxAgeDays;
  }
  if (Array.isArray(options.denyLanguages) && options.denyLanguages.length) {
    payload.deny_languages = options.denyLanguages;
  }
  return request('/api/discover', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function startEnrichment(config = {}) {
  const {
    mode,
    limit,
    options,
    scope,
    category,
    filters,
    forceRun = false,
    neverReenrich = false,
  } = config;

  const payload = {
    limit: limit ?? null,
    forceRun: Boolean(forceRun),
    neverReenrich: Boolean(neverReenrich),
  };
  if (mode) {
    payload.mode = mode;
  }
  if (options) {
    payload.options = options;
  }
  if (scope) {
    payload.scope = scope;
  }
  if (category) {
    payload.category = category;
  }
  if (filters) {
    payload.filters = filters;
  }

  return request('/api/enrich', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

export async function fetchStats() {
  return request('/api/stats');
}

export async function notifyDiscoveryLoopStart(state = {}) {
  return request('/api/discovery/loop/start', {
    method: 'POST',
    body: JSON.stringify(state),
  });
}

export async function notifyDiscoveryLoopProgress(state = {}) {
  return request('/api/discovery/loop/progress', {
    method: 'POST',
    body: JSON.stringify(state),
  });
}

export async function notifyDiscoveryLoopStop() {
  return request('/api/discovery/loop/stop', {
    method: 'POST',
    body: JSON.stringify({}),
  });
}

export async function notifyDiscoveryLoopComplete(state = {}) {
  return request('/api/discovery/loop/complete', {
    method: 'POST',
    body: JSON.stringify(state),
  });
}

export async function downloadCsv(category, filters, sort, order, options = {}) {
  const query = buildQuery(filters, category, sort, order, 10000, 0);
  const params = new URLSearchParams(query);
  if (options.archiveExported) {
    params.set('archive_exported', 'true');
  }
  const response = await fetch(`/api/export/csv?${params.toString()}`, {
    headers: {
      ...JSON_HEADERS,
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to export CSV');
  }
  const exportTimestamp = response.headers.get('x-export-timestamp');
  const csvText = await response.text();
  return { csv: csvText, exportTimestamp };
}

export async function downloadBundle() {
  const response = await fetch('/api/export/bundle', {
    headers: {
      Accept: 'application/zip, application/octet-stream',
    },
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Failed to export project bundle');
  }
  const exportTimestamp = response.headers.get('x-export-timestamp');
  const blob = await response.blob();
  return { blob, exportTimestamp };
}

export async function importBlacklist(file) {
  const formData = new FormData();
  formData.append('file', file);
  const response = await fetch('/api/blacklist/import', {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || 'Import failed');
  }
  return response.json();
}

export async function archiveExportedRows(exportedAt) {
  return request('/api/channels/archive_exported', {
    method: 'POST',
    body: JSON.stringify({ exported_at: exportedAt }),
  });
}
