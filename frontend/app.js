import {
  archiveChannels,
  archiveExportedRows,
  blacklistByFilter,
  blacklistChannels,
  discoverChannels,
  downloadCsv,
  downloadBundle,
  fetchChannels,
  fetchStats,
  importBlacklist,
  notifyDiscoveryLoopComplete,
  notifyDiscoveryLoopProgress,
  notifyDiscoveryLoopStart,
  notifyDiscoveryLoopStop,
  restoreByFilter,
  restoreChannels,
  startEnrichment,
} from './api.js';

const Category = {
  ACTIVE: 'active',
  ARCHIVED: 'archived',
  BLACKLISTED: 'blacklisted',
};

const CategoryInfo = {
  [Category.ACTIVE]: {
    icon: '🟢',
    label: 'Active',
    title: 'Active channels',
    subtitle: 'Fresh leads waiting for outreach.',
    primaryAction: { text: '🟡 Archive current filter', tone: 'yellow' },
    secondaryAction: { text: '🔴 Blacklist (filtered)', tone: 'red' },
  },
  [Category.ARCHIVED]: {
    icon: '🟡',
    label: 'Archived',
    title: 'Archived channels',
    subtitle: 'Previously contacted or paused leads.',
    primaryAction: { text: '🟢 Restore (filtered)', tone: 'green' },
    secondaryAction: { text: '🔴 Blacklist (filtered)', tone: 'red' },
  },
  [Category.BLACKLISTED]: {
    icon: '🔴',
    label: 'Blacklisted',
    title: 'Blacklisted channels',
    subtitle: 'Channels hidden from future discovery.',
    primaryAction: { text: '🟢 Restore (filtered)', tone: 'green' },
    secondaryAction: null,
  },
};

const StatusToneClass = {
  info: 'status-bar--info',
  success: 'status-bar--success',
  error: 'status-bar--error',
};

const StatusBadgeClass = {
  [Category.ACTIVE]: 'status-chip--success',
  [Category.ARCHIVED]: 'status-chip--warning',
  [Category.BLACKLISTED]: 'status-chip--danger',
};

const StatusBadgeIcon = {
  [Category.ACTIVE]: '🟢',
  [Category.ARCHIVED]: '🟡',
  [Category.BLACKLISTED]: '🔴',
};

function defaultFilters() {
  return {
    query: '',
    languages: [],
    statuses: [],
    minSubscribers: '',
    maxSubscribers: '',
    emailsOnly: false,
    uniqueEmails: false,
    emailGateOnly: false,
  };
}

function createTableState() {
  return {
    filters: defaultFilters(),
    sort: 'created_at',
    order: 'desc',
    page: 0,
    limit: 50,
    rows: [],
    total: 0,
    loading: false,
    error: null,
  };
}

function formatNumber(value) {
  if (value == null) {
    return '—';
  }
  return new Intl.NumberFormat().format(value);
}

function formatDate(value) {
  if (!value) {
    return '—';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleDateString();
}

function formatDateTime(value) {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date.toLocaleString();
}

function parseList(value) {
  if (!value) {
    return [];
  }
  return value
    .split(/[\s,;]+/)
    .map((entry) => entry.trim().toLowerCase())
    .filter(Boolean);
}

function escapeHtml(value) {
  if (value == null) {
    return '';
  }
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const DISCOVERY_SETTINGS_STORAGE_KEY = 'dashboard:discoverySettings';
const DEFAULT_DISCOVERY_KEYWORDS =
  'crypto, bitcoin, ethereum, defi, altcoin, memecoin, onchain, crypto trading';
const ENRICHMENT_SETTINGS_STORAGE_KEY = 'dashboard:enrichmentSettings';

const DEFAULT_EMAIL_MODE = 'channel_and_videos';
const DEFAULT_LANGUAGE_MODE = 'precise';

function parseDiscoveryKeywords(text) {
  if (!text) {
    return [];
  }
  return text
    .split(/[\n,]+/)
    .map((word) => word.trim())
    .filter(Boolean);
}

function parseDiscoveryDenyLanguages(text) {
  if (!text) {
    return [];
  }
  return text
    .split(/[\s,]+/)
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
}

function defaultEnrichmentOptions() {
  return {
    emails_from_channel: true,
    emails_from_videos: true,
    language_basic: true,
    language_precise: true,
    update_metadata: true,
    update_activity: true,
  };
}

function emailOnlyEnrichmentOptions() {
  return {
    emails_from_channel: true,
    emails_from_videos: false,
    language_basic: false,
    language_precise: false,
    update_metadata: false,
    update_activity: false,
  };
}

function defaultEnrichmentSettingsState() {
  return {
    scope: 'filtered',
    emailMode: DEFAULT_EMAIL_MODE,
    languageMode: DEFAULT_LANGUAGE_MODE,
    options: defaultEnrichmentOptions(),
  };
}

function emailOptionsFromMode(mode = DEFAULT_EMAIL_MODE) {
  const normalized = String(mode || DEFAULT_EMAIL_MODE).trim().toLowerCase();
  switch (normalized) {
    case 'off':
      return { emails_from_channel: false, emails_from_videos: false };
    case 'channel_only':
      return { emails_from_channel: true, emails_from_videos: false };
    case 'videos_only':
      return { emails_from_channel: false, emails_from_videos: true };
    case 'channel_and_videos':
    default:
      return { emails_from_channel: true, emails_from_videos: true };
  }
}

function languageOptionsFromMode(mode = DEFAULT_LANGUAGE_MODE) {
  const normalized = String(mode || DEFAULT_LANGUAGE_MODE).trim().toLowerCase();
  switch (normalized) {
    case 'off':
      return { language_basic: false, language_precise: false };
    case 'fast':
      return { language_basic: true, language_precise: false };
    case 'precise':
    default:
      return { language_basic: true, language_precise: true };
  }
}

function deriveEmailModeFromOptions(options = defaultEnrichmentOptions()) {
  if (!options.emails_from_channel && !options.emails_from_videos) {
    return 'off';
  }
  if (options.emails_from_channel && options.emails_from_videos) {
    return 'channel_and_videos';
  }
  if (options.emails_from_channel) {
    return 'channel_only';
  }
  if (options.emails_from_videos) {
    return 'videos_only';
  }
  return DEFAULT_EMAIL_MODE;
}

function deriveLanguageModeFromOptions(options = defaultEnrichmentOptions()) {
  if (!options.language_basic && !options.language_precise) {
    return 'off';
  }
  if (options.language_precise) {
    return 'precise';
  }
  if (options.language_basic) {
    return 'fast';
  }
  return DEFAULT_LANGUAGE_MODE;
}

function normalizeEnrichmentOptions(raw, fallback = defaultEnrichmentOptions()) {
  if (!raw || typeof raw !== 'object') {
    return { ...fallback };
  }
  return {
    emails_from_channel: Boolean(raw.emails_from_channel ?? fallback.emails_from_channel),
    emails_from_videos: Boolean(raw.emails_from_videos ?? fallback.emails_from_videos),
    language_basic: Boolean(raw.language_basic ?? fallback.language_basic),
    language_precise: Boolean(raw.language_precise ?? fallback.language_precise),
    update_metadata: Boolean(raw.update_metadata ?? fallback.update_metadata),
    update_activity: Boolean(raw.update_activity ?? fallback.update_activity),
  };
}

function describeEnrichmentOptions(options) {
  const parts = [];
  if (options.emails_from_channel && options.emails_from_videos) {
    parts.push('emails (channel + videos)');
  } else if (options.emails_from_videos) {
    parts.push('emails from videos');
  } else if (options.emails_from_channel) {
    parts.push('emails from channel');
  }
  if (options.language_precise && options.language_basic) {
    parts.push('precise language');
  } else if (options.language_precise) {
    parts.push('precise language');
  } else if (options.language_basic) {
    parts.push('basic language');
  }
  if (options.update_metadata || options.update_activity) {
    const misc = [];
    if (options.update_metadata) {
      misc.push('metadata');
    }
    if (options.update_activity) {
      misc.push('activity');
    }
    if (misc.length) {
      parts.push(misc.join(' + '));
    }
  }
  if (!parts.length) {
    return 'Custom (no steps enabled)';
  }
  return parts.join(', ');
}

function defaultDiscoverySettingsState() {
  return {
    keywordsText: DEFAULT_DISCOVERY_KEYWORDS,
    keywords: parseDiscoveryKeywords(DEFAULT_DISCOVERY_KEYWORDS),
    perKeyword: 5,
    lastUploadMaxAgeDays: null,
    denyLanguagesText: '',
    denyLanguages: [],
    autoEnrichEnabled: false,
    autoEnrichMode: 'email_only',
    autoEnrichOptions: emailOnlyEnrichmentOptions(),
    enrichLimit: null,
    runUntilStopped: false,
  };
}

class Dashboard {
  constructor(root) {
    this.root = root;
    this.activeTab = Category.ACTIVE;
    this.tables = {
      [Category.ACTIVE]: createTableState(),
      [Category.ARCHIVED]: createTableState(),
      [Category.BLACKLISTED]: createTableState(),
    };
    this.stats = null;
    this.statusTimeout = null;
    this.eventSource = null;
    this.enrichmentBusy = false;
    this.filterDebounce = null;
    this.discoveryLoopActive = false;
    this.discoveryLoopStopping = false;
    this.discoveryLoopStats = { runs: 0, found: 0 };
    this.pendingAutoEnrich = 0;
    this.autoEnrichQueuedNotified = false;
    this.discoverySettings = defaultDiscoverySettingsState();
    this.enrichmentSettings = defaultEnrichmentSettingsState();
    this.activeDropdown = null;
    this.activeDropdownToggle = null;
    this.handleDocumentClick = this.handleDocumentClick.bind(this);
    this.handleDocumentKeydown = this.handleDocumentKeydown.bind(this);
    this.statsPromise = null;
    this.statsPollTimer = null;
    this.lastLoopCompletionVersion = null;
    this.discoveryLoopCompletionPayload = null;
    this.finalizingDiscoveryLoop = false;
    this.discoveryLoopCompletionSent = false;
  }

  init() {
    this.cacheElements();
    this.loadDiscoverySettings();
    this.loadEnrichmentSettings();
    this.bindEvents();
    this.renderTabs();
    this.applyFiltersToUI(this.tables[this.activeTab].filters);
    this.updateSortInputs();
    this.updatePagination();
    this.renderTable(true);
    this.updateDiscoverySummary();
    this.updateQuickStats();
    this.updateStatusBar('', 'info');
    this.updateSystemStatus(null);
    this.loadStats();
    this.loadTable(this.activeTab);
    this.startStatsPolling();
    this.updateEnrichmentLastUsed();
  }

  cacheElements() {
    this.el = {
      quickStats: this.root.querySelector('#quickStats'),
      helpBtn: this.root.querySelector('#helpBtn'),
      helpModal: document.getElementById('helpModal'),
      helpModalClose: document.getElementById('helpModalClose'),
      tabButtons: Array.from(this.root.querySelectorAll('[data-category]')),
      tableTitle: this.root.querySelector('#tableTitle'),
      tableSubtitle: this.root.querySelector('#tableSubtitle'),
      bulkPrimaryBtn: this.root.querySelector('#bulkPrimaryBtn'),
      bulkSecondaryBtn: this.root.querySelector('#bulkSecondaryBtn'),
      exportCsvBtn: this.root.querySelector('#exportCsvBtn'),
      exportBundleBtn: this.root.querySelector('#exportBundleBtn'),
      importExportToggle: this.root.querySelector('#importExportToggle'),
      importExportMenu: this.root.querySelector('#importExportMenu'),
      sortSelect: this.root.querySelector('#sortSelect'),
      orderSelect: this.root.querySelector('#orderSelect'),
      tableBody: this.root.querySelector('#tableBody'),
      tableStatus: this.root.querySelector('#tableStatus'),
      paginationLabel: this.root.querySelector('#paginationLabel'),
      prevPageBtn: this.root.querySelector('#prevPageBtn'),
      nextPageBtn: this.root.querySelector('#nextPageBtn'),
      statusBar: this.root.querySelector('#statusBar'),
      progressBar: this.root.querySelector('#progressBar'),
      summaryBar: this.root.querySelector('#summaryBar'),
      systemStatusBar: this.root.querySelector('#systemStatusBar'),
      discoverKeywords: document.getElementById('discoverKeywords'),
      discoverPerKeyword: document.getElementById('discoverPerKeyword'),
      discoverDenyLanguages: document.getElementById('discoverDenyLanguages'),
      discoverLastUploadMaxAge: document.getElementById('discoverLastUploadMaxAge'),
      discoverBtn: this.root.querySelector('#discoverBtn'),
      discoverSummary: this.root.querySelector('#discoverSummary'),
      discoverStopBtn: this.root.querySelector('#discoverStopBtn'),
      discoverRunCounter: this.root.querySelector('#discoverRunCounter'),
      discoverAutoEnrichToggle: document.getElementById('discoverAutoEnrichToggle'),
      discoverAutoEnrichMode: document.getElementById('discoverAutoEnrichMode'),
      discoverRunUntilStopped: document.getElementById('discoverRunUntilStopped'),
      discoverSettingsModal: document.getElementById('discoverSettingsModal'),
      discoverSettingsClose: document.getElementById('discoverSettingsCloseBtn'),
      discoverSettingsCancel: document.getElementById('discoverSettingsCancelBtn'),
      discoverSettingsForm: document.getElementById('discoverSettingsForm'),
      discoverSettingsStart: document.getElementById('discoverSettingsStartBtn'),
      enrichBtn: this.root.querySelector('#enrichBtn'),
      enrichLastUsed: this.root.querySelector('#enrichLastUsed'),
      enrichSettingsModal: document.getElementById('enrichSettingsModal'),
      enrichSettingsClose: document.getElementById('enrichSettingsCloseBtn'),
      enrichSettingsCancel: document.getElementById('enrichSettingsCancelBtn'),
      enrichSettingsForm: document.getElementById('enrichSettingsForm'),
      enrichScopeFiltered: document.getElementById('enrichScopeFiltered'),
      enrichScopeAll: document.getElementById('enrichScopeAllActive'),
      enrichEmailsEnabled: document.getElementById('enrichEmailsEnabled'),
      enrichEmailsMode: document.getElementById('enrichEmailsMode'),
      enrichLanguageEnabled: document.getElementById('enrichLanguageEnabled'),
      enrichLanguageMode: document.getElementById('enrichLanguageMode'),
      enrichUpdateMetadata: document.getElementById('enrichUpdateMetadata'),
      enrichUpdateActivity: document.getElementById('enrichUpdateActivity'),
      enrichLimit: document.getElementById('enrichLimit'),
      enrichForceToggle: this.root.querySelector('#enrichForceToggle'),
      enrichNeverToggle: this.root.querySelector('#enrichNeverToggle'),
      importBlacklistBtn: this.root.querySelector('#importBlacklistBtn'),
      modal: document.getElementById('blacklistModal'),
      modalClose: document.getElementById('blacklistCloseBtn'),
      modalCancel: document.getElementById('blacklistCancelBtn'),
      modalForm: document.getElementById('blacklistForm'),
      modalFile: document.getElementById('blacklistFile'),
      modalSummary: document.getElementById('blacklistSummary'),
      modalSubmit: document.getElementById('blacklistSubmitBtn'),
      filterQuery: this.root.querySelector('#filterQuery'),
      filterLanguages: this.root.querySelector('#filterLanguages'),
      filterMinSubs: this.root.querySelector('#filterMinSubs'),
      filterMaxSubs: this.root.querySelector('#filterMaxSubs'),
      filterEmailsOnly: this.root.querySelector('#filterEmailsOnly'),
      filterUniqueEmails: this.root.querySelector('#filterUniqueEmails'),
      filterEmailGateOnly: this.root.querySelector('#filterEmailGateOnly'),
      filterStatusCheckboxes: Array.from(this.root.querySelectorAll('input[name="statusFilter"]')),
    };
  }

  bindEvents() {
    this.el.tabButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const category = button.dataset.category;
        if (category && category !== this.activeTab) {
          this.changeTab(category);
        }
      });
    });

    this.el.helpBtn?.addEventListener('click', () => this.openHelpModal());
    this.el.helpModalClose?.addEventListener('click', () => this.closeHelpModal());
    this.el.helpModal?.addEventListener('click', (event) => {
      if (event.target === this.el.helpModal) {
        this.closeHelpModal();
      }
    });

    this.el.sortSelect.addEventListener('change', () => {
      this.updateSort(this.el.sortSelect.value);
    });
    this.el.orderSelect.addEventListener('change', () => {
      this.updateOrder(this.el.orderSelect.value);
    });

    const scheduleFilters = () => {
      clearTimeout(this.filterDebounce);
      this.filterDebounce = setTimeout(() => {
        this.applyFiltersFromUI();
      }, 250);
    };

    this.el.filterQuery.addEventListener('input', scheduleFilters);
    this.el.filterLanguages.addEventListener('change', scheduleFilters);
    this.el.filterMinSubs.addEventListener('change', scheduleFilters);
    this.el.filterMaxSubs.addEventListener('change', scheduleFilters);
    this.el.filterEmailsOnly.addEventListener('change', () => {
      this.el.filterUniqueEmails.disabled = !this.el.filterEmailsOnly.checked;
      if (!this.el.filterEmailsOnly.checked) {
        this.el.filterUniqueEmails.checked = false;
      }
      scheduleFilters();
    });
    this.el.filterUniqueEmails.addEventListener('change', scheduleFilters);
    this.el.filterEmailGateOnly.addEventListener('change', scheduleFilters);
    this.el.filterStatusCheckboxes.forEach((checkbox) => {
      checkbox.addEventListener('change', scheduleFilters);
    });

    this.el.prevPageBtn.addEventListener('click', () => this.changePage(-1));
    this.el.nextPageBtn.addEventListener('click', () => this.changePage(1));

    this.el.tableBody.addEventListener('click', (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      const button = target.closest('button[data-action]');
      if (!button) {
        return;
      }
      const action = button.dataset.action;
      const channelId = button.dataset.id;
      if (!action || !channelId) {
        return;
      }
      this.handleRowAction(action, channelId);
    });

    this.el.bulkPrimaryBtn.addEventListener('click', () => this.handleBulkPrimary());
    this.el.bulkSecondaryBtn.addEventListener('click', () => this.handleBulkSecondary());
    this.el.discoverBtn.addEventListener('click', () => this.openDiscoverSettings());
    this.el.discoverStopBtn?.addEventListener('click', () => this.stopDiscoveryLoop());

    this.el.importExportToggle?.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      this.toggleDropdown(this.el.importExportToggle, this.el.importExportMenu);
    });
    this.el.importExportMenu?.addEventListener('click', (event) => {
      event.stopPropagation();
    });
    this.el.exportCsvBtn?.addEventListener('click', (event) => {
      event.stopPropagation();
      this.closeDropdown(this.el.importExportMenu, this.el.importExportToggle);
      this.handleExportCsv();
    });
    this.el.exportBundleBtn?.addEventListener('click', (event) => {
      event.stopPropagation();
      this.closeDropdown(this.el.importExportMenu, this.el.importExportToggle);
      this.handleExportBundle();
    });

    this.el.enrichBtn.addEventListener('click', (event) => {
      event.preventDefault();
      this.openEnrichmentSettings();
    });
    this.el.enrichSettingsClose?.addEventListener('click', () => this.closeEnrichmentSettings());
    this.el.enrichSettingsCancel?.addEventListener('click', () => this.closeEnrichmentSettings());
    this.el.enrichSettingsModal?.addEventListener('click', (event) => {
      if (event.target === this.el.enrichSettingsModal) {
        this.closeEnrichmentSettings();
      }
    });
    this.el.enrichSettingsForm?.addEventListener('submit', async (event) => {
      event.preventDefault();
      await this.handleEnrichmentSettingsSubmit();
    });

    this.el.enrichEmailsEnabled?.addEventListener('change', () =>
      this.updateEnrichmentModeControls(),
    );
    this.el.enrichLanguageEnabled?.addEventListener('change', () =>
      this.updateEnrichmentModeControls(),
    );

    this.el.discoverAutoEnrichToggle?.addEventListener('change', () => {
      if (this.el.discoverAutoEnrichMode) {
        this.el.discoverAutoEnrichMode.disabled = !this.el.discoverAutoEnrichToggle.checked;
      }
    });

    this.el.discoverSettingsClose?.addEventListener('click', () => this.closeDiscoverSettings());
    this.el.discoverSettingsCancel?.addEventListener('click', () => this.closeDiscoverSettings());
    this.el.discoverSettingsModal?.addEventListener('click', (event) => {
      if (event.target === this.el.discoverSettingsModal) {
        this.closeDiscoverSettings();
      }
    });
    this.el.discoverSettingsForm?.addEventListener('submit', async (event) => {
      event.preventDefault();
      await this.handleDiscoverSettingsSubmit();
    });

    this.el.importBlacklistBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      this.closeDropdown(this.el.importExportMenu, this.el.importExportToggle);
      this.openModal();
    });
    this.el.modalClose.addEventListener('click', () => this.closeModal());
    this.el.modalCancel.addEventListener('click', () => this.closeModal());
    this.el.modal.addEventListener('click', (event) => {
      if (event.target === this.el.modal) {
        this.closeModal();
      }
    });

    this.el.modalForm.addEventListener('submit', (event) => {
      event.preventDefault();
      this.handleBlacklistImport();
    });

    document.addEventListener('click', this.handleDocumentClick);
    document.addEventListener('keydown', this.handleDocumentKeydown);
  }

  loadDiscoverySettings() {
    const defaults = defaultDiscoverySettingsState();
    let settings = defaults;
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const stored = window.localStorage.getItem(DISCOVERY_SETTINGS_STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored);
          settings = this.normalizeDiscoverySettings({ ...defaults, ...parsed });
        }
      }
    } catch (error) {
      console.warn('Failed to load discovery settings', error);
      settings = defaults;
    }
    this.discoverySettings = settings;
  }

  loadEnrichmentSettings() {
    const defaults = defaultEnrichmentSettingsState();
    let settings = defaults;
    try {
      if (typeof window !== 'undefined' && window.localStorage) {
        const stored = window.localStorage.getItem(ENRICHMENT_SETTINGS_STORAGE_KEY);
        if (stored) {
          const parsed = JSON.parse(stored);
          settings = this.normalizeEnrichmentSettings({ ...defaults, ...parsed });
        }
      }
    } catch (error) {
      console.warn('Failed to load enrichment settings', error);
      settings = defaults;
    }
    this.enrichmentSettings = settings;
  }

  saveDiscoverySettings() {
    if (typeof window === 'undefined' || !window.localStorage) {
      return;
    }
    try {
      window.localStorage.setItem(
        DISCOVERY_SETTINGS_STORAGE_KEY,
        JSON.stringify(this.discoverySettings),
      );
    } catch (error) {
      console.warn('Failed to save discovery settings', error);
    }
  }

  normalizeDiscoverySettings(raw) {
    const defaults = defaultDiscoverySettingsState();
    const keywordsText =
      typeof raw.keywordsText === 'string'
        ? raw.keywordsText
        : Array.isArray(raw.keywords)
        ? raw.keywords.join(', ')
        : defaults.keywordsText;
    const keywords = parseDiscoveryKeywords(keywordsText);
    const perKeywordRaw = Number(raw.perKeyword);
    const perKeyword =
      Number.isNaN(perKeywordRaw) || perKeywordRaw <= 0
        ? defaults.perKeyword
        : Math.max(1, Math.floor(perKeywordRaw));

    let lastUploadMaxAgeDays = null;
    const lastRaw = raw.lastUploadMaxAgeDays;
    if (typeof lastRaw === 'number') {
      if (lastRaw > 0) {
        lastUploadMaxAgeDays = Math.floor(lastRaw);
      }
    } else if (typeof lastRaw === 'string' && lastRaw.trim() !== '') {
      const parsed = Number(lastRaw);
      if (!Number.isNaN(parsed) && parsed > 0) {
        lastUploadMaxAgeDays = Math.floor(parsed);
      }
    }

    const denyLanguagesText =
      typeof raw.denyLanguagesText === 'string'
        ? raw.denyLanguagesText
        : Array.isArray(raw.denyLanguages)
        ? raw.denyLanguages.join(', ')
        : '';
    const denyLanguages = parseDiscoveryDenyLanguages(denyLanguagesText);

    let enrichLimit = null;
    const enrichRaw = raw.enrichLimit;
    if (typeof enrichRaw === 'number') {
      if (enrichRaw > 0) {
        enrichLimit = Math.floor(enrichRaw);
      }
    } else if (typeof enrichRaw === 'string' && enrichRaw.trim() !== '') {
      const parsed = Number(enrichRaw);
      if (!Number.isNaN(parsed) && parsed > 0) {
        enrichLimit = Math.floor(parsed);
      }
    }

    const autoEnrichMode = raw.autoEnrichMode === 'full' ? 'full' : 'email_only';
    const autoEnrichEnabled = Boolean(raw.autoEnrichEnabled);
    const autoEnrichFallback =
      autoEnrichMode === 'full' ? defaultEnrichmentOptions() : emailOnlyEnrichmentOptions();
    const autoEnrichOptions = normalizeEnrichmentOptions(raw.autoEnrichOptions, autoEnrichFallback);
    const runUntilStopped = Boolean(raw.runUntilStopped);

    return {
      keywordsText,
      keywords,
      perKeyword,
      lastUploadMaxAgeDays,
      denyLanguagesText,
      denyLanguages,
      enrichLimit,
      autoEnrichEnabled,
      autoEnrichMode,
      autoEnrichOptions,
      runUntilStopped,
    };
  }

  normalizeEnrichmentSettings(raw) {
    const defaults = defaultEnrichmentSettingsState();
    const scope = raw.scope === 'all_active' ? 'all_active' : 'filtered';
    const options = normalizeEnrichmentOptions(raw.options, defaults.options);
    const emailMode =
      typeof raw.emailMode === 'string'
        ? raw.emailMode
        : typeof raw.email_mode === 'string'
        ? raw.email_mode
        : deriveEmailModeFromOptions(options);
    const languageMode =
      typeof raw.languageMode === 'string'
        ? raw.languageMode
        : typeof raw.language_mode === 'string'
        ? raw.language_mode
        : deriveLanguageModeFromOptions(options);

    const mergedOptions = {
      ...options,
      ...emailOptionsFromMode(emailMode),
      ...languageOptionsFromMode(languageMode),
    };

    return {
      scope,
      emailMode: emailMode || DEFAULT_EMAIL_MODE,
      languageMode: languageMode || DEFAULT_LANGUAGE_MODE,
      options: mergedOptions,
    };
  }

  saveEnrichmentSettings() {
    if (typeof window === 'undefined' || !window.localStorage) {
      return;
    }
    try {
      window.localStorage.setItem(
        ENRICHMENT_SETTINGS_STORAGE_KEY,
        JSON.stringify(this.enrichmentSettings),
      );
    } catch (error) {
      console.warn('Failed to save enrichment settings', error);
    }
  }

  setDiscoverySettings(settings) {
    const normalized = this.normalizeDiscoverySettings({
      ...this.discoverySettings,
      ...settings,
    });
    this.discoverySettings = normalized;
    if (!normalized.autoEnrichEnabled) {
      this.pendingAutoEnrich = 0;
      this.autoEnrichQueuedNotified = false;
    }
    this.saveDiscoverySettings();
    this.updateDiscoverySummary();
  }

  updateDiscoverySummary() {
    if (!this.el.discoverSummary) {
      return;
    }
    const settings = this.discoverySettings || defaultDiscoverySettingsState();
    const parts = [];
    if (settings.keywords?.length) {
      const preview = settings.keywords.slice(0, 3).join(', ');
      const hasMore = settings.keywords.length > 3;
      parts.push(`Keywords: ${preview}${hasMore ? '…' : ''}`);
    } else {
      parts.push('Keywords: not set');
    }
    parts.push(`Per keyword: ${formatNumber(settings.perKeyword || 5)}`);
    if (settings.lastUploadMaxAgeDays != null) {
      parts.push(`Max age: ${formatNumber(settings.lastUploadMaxAgeDays)} days`);
    }
    if (settings.denyLanguages?.length) {
      parts.push(`Deny languages: ${settings.denyLanguages.join(', ')}`);
    }
    if (settings.enrichLimit != null) {
      parts.push(`Enrich limit: ${formatNumber(settings.enrichLimit)}`);
    }
    if (settings.autoEnrichEnabled) {
      const description = describeEnrichmentOptions(
        settings.autoEnrichOptions ||
          (settings.autoEnrichMode === 'full'
            ? defaultEnrichmentOptions()
            : emailOnlyEnrichmentOptions()),
      );
      parts.push(`Auto enrich: ${description}`);
    } else {
      parts.push('Auto enrich: off');
    }
    if (settings.runUntilStopped) {
      parts.push('Loop: on');
    }
    this.el.discoverSummary.textContent = parts.join(' · ');
  }

  updateEnrichmentLastUsed() {
    if (!this.el.enrichLastUsed) {
      return;
    }
    const settings = this.enrichmentSettings || defaultEnrichmentSettingsState();
    const description = describeEnrichmentOptions(settings.options);
    this.el.enrichLastUsed.textContent = `Last used: ${description}`;
  }

  populateEnrichmentSettingsForm() {
    const settings = this.enrichmentSettings || defaultEnrichmentSettingsState();
    if (this.el.enrichScopeFiltered) {
      this.el.enrichScopeFiltered.checked = settings.scope !== 'all_active';
    }
    if (this.el.enrichScopeAll) {
      this.el.enrichScopeAll.checked = settings.scope === 'all_active';
    }
    if (this.el.enrichEmailsEnabled) {
      this.el.enrichEmailsEnabled.checked = settings.emailMode !== 'off';
    }
    if (this.el.enrichEmailsMode) {
      this.el.enrichEmailsMode.value = settings.emailMode || DEFAULT_EMAIL_MODE;
    }
    if (this.el.enrichLanguageEnabled) {
      this.el.enrichLanguageEnabled.checked = settings.languageMode !== 'off';
    }
    if (this.el.enrichLanguageMode) {
      this.el.enrichLanguageMode.value = settings.languageMode || DEFAULT_LANGUAGE_MODE;
    }
    if (this.el.enrichUpdateMetadata) {
      this.el.enrichUpdateMetadata.checked = Boolean(settings.options.update_metadata);
    }
    if (this.el.enrichUpdateActivity) {
      this.el.enrichUpdateActivity.checked = Boolean(settings.options.update_activity);
    }
    this.updateEnrichmentModeControls();
  }

  updateEnrichmentModeControls() {
    if (this.el.enrichEmailsMode) {
      this.el.enrichEmailsMode.disabled = !this.el.enrichEmailsEnabled?.checked;
      this.el.enrichEmailsMode.closest('.modal-select')?.classList.toggle(
        'modal-select--disabled',
        !this.el.enrichEmailsEnabled?.checked,
      );
    }
    if (this.el.enrichLanguageMode) {
      this.el.enrichLanguageMode.disabled = !this.el.enrichLanguageEnabled?.checked;
      this.el.enrichLanguageMode.closest('.modal-select')?.classList.toggle(
        'modal-select--disabled',
        !this.el.enrichLanguageEnabled?.checked,
      );
    }
  }

  openEnrichmentSettings() {
    if (!this.el.enrichSettingsModal) {
      return;
    }
    this.closeAllDropdowns();
    this.populateEnrichmentSettingsForm();
    this.el.enrichSettingsModal.hidden = false;
    this.el.enrichSettingsModal.setAttribute('aria-hidden', 'false');
  }

  closeEnrichmentSettings() {
    if (!this.el.enrichSettingsModal) {
      return;
    }
    this.el.enrichSettingsModal.hidden = true;
    this.el.enrichSettingsModal.setAttribute('aria-hidden', 'true');
  }

  openDiscoverSettings() {
    if (!this.el.discoverSettingsModal) {
      return;
    }
    this.closeAllDropdowns();
    if (this.discoveryLoopActive) {
      this.updateStatusBar(
        'Discovery loop is running. Stop it before changing settings.',
        'info',
      );
      return;
    }
    this.populateDiscoverSettingsForm();
    this.el.discoverSettingsModal.removeAttribute('hidden');
  }

  closeDiscoverSettings() {
    this.el.discoverSettingsModal?.setAttribute('hidden', 'true');
  }

  populateDiscoverSettingsForm() {
    const settings = this.discoverySettings || defaultDiscoverySettingsState();
    if (this.el.discoverKeywords) {
      this.el.discoverKeywords.value = settings.keywordsText || '';
    }
    if (this.el.discoverPerKeyword) {
      this.el.discoverPerKeyword.value = String(settings.perKeyword ?? 5);
    }
    if (this.el.discoverLastUploadMaxAge) {
      this.el.discoverLastUploadMaxAge.value =
        settings.lastUploadMaxAgeDays != null ? String(settings.lastUploadMaxAgeDays) : '';
    }
    if (this.el.discoverDenyLanguages) {
      this.el.discoverDenyLanguages.value = settings.denyLanguagesText || '';
    }
    if (this.el.discoverRunUntilStopped) {
      this.el.discoverRunUntilStopped.checked = Boolean(settings.runUntilStopped);
    }
    if (this.el.discoverAutoEnrichToggle) {
      this.el.discoverAutoEnrichToggle.checked = Boolean(settings.autoEnrichEnabled);
    }
    if (this.el.discoverAutoEnrichMode) {
      this.el.discoverAutoEnrichMode.value =
        settings.autoEnrichMode === 'full' ? 'full' : 'email_only';
      this.el.discoverAutoEnrichMode.disabled = !settings.autoEnrichEnabled;
    }
    if (this.el.enrichLimit) {
      this.el.enrichLimit.value =
        settings.enrichLimit != null ? String(settings.enrichLimit) : '';
    }
  }

  collectDiscoverySettingsFromDialog() {
    if (!this.el.discoverKeywords || !this.el.discoverPerKeyword) {
      return null;
    }
    const keywordsText = this.el.discoverKeywords.value;
    const keywords = parseDiscoveryKeywords(keywordsText);
    if (!keywords.length) {
      this.updateStatusBar('Please provide at least one keyword.', 'error');
      return null;
    }

    const perKeywordRaw = Number(this.el.discoverPerKeyword.value);
    const perKeyword =
      Number.isNaN(perKeywordRaw) || perKeywordRaw <= 0
        ? this.discoverySettings?.perKeyword || 5
        : Math.max(1, Math.floor(perKeywordRaw));

    const denyLanguagesText = this.el.discoverDenyLanguages?.value ?? '';
    const denyLanguages = parseDiscoveryDenyLanguages(denyLanguagesText);

    const maxAgeRaw = this.el.discoverLastUploadMaxAge?.value ?? '';
    let lastUploadMaxAgeDays = null;
    if (typeof maxAgeRaw === 'string' && maxAgeRaw.trim() !== '') {
      const parsed = Number(maxAgeRaw);
      if (Number.isNaN(parsed) || parsed < 0) {
        this.updateStatusBar('Last upload max age must be zero or greater.', 'error');
        return null;
      }
      lastUploadMaxAgeDays = parsed > 0 ? Math.floor(parsed) : null;
    }

    const enrichLimitRaw = this.el.enrichLimit?.value ?? '';
    let enrichLimit = null;
    if (typeof enrichLimitRaw === 'string' && enrichLimitRaw.trim() !== '') {
      const parsed = Number(enrichLimitRaw);
      if (Number.isNaN(parsed) || parsed <= 0) {
        this.updateStatusBar('Enrichment limit must be a positive number.', 'error');
        return null;
      }
      enrichLimit = Math.floor(parsed);
    }

    const autoEnrichEnabled = Boolean(this.el.discoverAutoEnrichToggle?.checked);
    const autoEnrichMode =
      this.el.discoverAutoEnrichMode?.value === 'full' ? 'full' : 'email_only';
    const autoEnrichOptions =
      autoEnrichMode === 'full' ? defaultEnrichmentOptions() : emailOnlyEnrichmentOptions();
    const runUntilStopped = Boolean(this.el.discoverRunUntilStopped?.checked);

    return {
      keywordsText,
      keywords,
      perKeyword,
      denyLanguagesText,
      denyLanguages,
      lastUploadMaxAgeDays,
      enrichLimit,
      autoEnrichEnabled,
      autoEnrichMode,
      autoEnrichOptions,
      runUntilStopped,
    };
  }

  collectDiscoveryInputsFromSettings() {
    const settings = this.discoverySettings;
    if (!settings || !settings.keywords || !settings.keywords.length) {
      this.updateStatusBar('Please provide at least one keyword.', 'error');
      return null;
    }
    return {
      keywords: [...settings.keywords],
      perKeyword: Math.max(1, Math.floor(Number(settings.perKeyword) || 1)),
      denyLanguages: Array.isArray(settings.denyLanguages)
        ? [...settings.denyLanguages]
        : [],
      lastUploadMaxAgeDays: settings.lastUploadMaxAgeDays,
    };
  }

  async handleDiscoverSettingsSubmit() {
    if (this.discoveryLoopActive) {
      this.updateStatusBar(
        'Discovery loop is running. Stop it before starting a manual discovery.',
        'info',
      );
      return;
    }
    const settings = this.collectDiscoverySettingsFromDialog();
    if (!settings) {
      return;
    }
    this.setDiscoverySettings(settings);
    this.closeDiscoverSettings();
    const inputs = this.collectDiscoveryInputsFromSettings();
    if (!inputs) {
      return;
    }
    if (settings.runUntilStopped) {
      await this.startDiscoveryLoop(inputs);
    } else {
      await this.performDiscoveryRun(inputs);
    }
  }

  renderTabs() {
    this.el.tabButtons.forEach((button) => {
      const category = button.dataset.category;
      if (!category) {
        return;
      }
      const info = CategoryInfo[category];
      if (!info) {
        return;
      }
      button.innerHTML = `${info.icon} ${info.label} <span class="tab-count" data-count-for="${category}">(0)</span>`;
      button.classList.toggle('is-active', category === this.activeTab);
    });
  }

  applyFiltersToUI(filters) {
    this.el.filterQuery.value = filters.query || '';
    this.el.filterLanguages.value = filters.languages.join(', ');
    this.el.filterMinSubs.value = filters.minSubscribers ?? '';
    this.el.filterMaxSubs.value = filters.maxSubscribers ?? '';
    this.el.filterEmailsOnly.checked = Boolean(filters.emailsOnly);
    this.el.filterUniqueEmails.checked = Boolean(filters.uniqueEmails);
    this.el.filterUniqueEmails.disabled = !filters.emailsOnly;
    this.el.filterEmailGateOnly.checked = Boolean(filters.emailGateOnly);
    this.el.filterStatusCheckboxes.forEach((checkbox) => {
      checkbox.checked = filters.statuses.includes(checkbox.value);
    });
  }

  updateSortInputs() {
    const table = this.tables[this.activeTab];
    this.el.sortSelect.value = table.sort;
    this.el.orderSelect.value = table.order;
  }

  updateQuickStats() {
    if (!this.el.quickStats) {
      return;
    }
    if (!this.stats) {
      this.el.quickStats.innerHTML = '<span class="quick-stats__loading">Loading stats…</span>';
      return;
    }
    const active = formatNumber(this.stats[Category.ACTIVE] || 0);
    const archived = formatNumber(this.stats[Category.ARCHIVED] || 0);
    const blacklisted = formatNumber(this.stats[Category.BLACKLISTED] || 0);
    const unique = formatNumber(this.stats.unique_emails || 0);
    this.el.quickStats.innerHTML = `
      <span>🟢 Active: ${active}</span>
      <span>🟡 Archived: ${archived}</span>
      <span>🔴 Blacklisted: ${blacklisted}</span>
      <span>Unique emails: ${unique}</span>
    `;
    this.el.tabButtons.forEach((button) => {
      const category = button.dataset.category;
      const countEl = button.querySelector('[data-count-for]');
      if (countEl && category) {
        countEl.textContent = `(${formatNumber(this.stats[category] || 0)})`;
      }
    });
  }

  startStatsPolling() {
    if (this.statsPollTimer) {
      return;
    }
    this.statsPollTimer = setInterval(() => {
      this.loadStats();
    }, 5000);
  }

  renderSystemStatusItem(indicatorClass, text, options = {}) {
    if (!text) {
      return '';
    }
    const classes = ['system-status__item'];
    if (options.muted) {
      classes.push('system-status__item--muted');
    }
    const indicator = indicatorClass || 'system-status__indicator--idle';
    return `
      <span class="${classes.join(' ')}">
        <span class="system-status__indicator ${indicator}"></span>
        <span>${escapeHtml(text)}</span>
      </span>
    `;
  }

  updateSystemStatus(stats, options = {}) {
    if (!this.el.systemStatusBar) {
      return;
    }
    if (!stats) {
      this.el.systemStatusBar.innerHTML = this.renderSystemStatusItem(
        'system-status__indicator--idle',
        options.unavailable ? 'System status unavailable' : 'Collecting system status…',
        { muted: true }
      );
      return;
    }
    const loop = stats.discoveryLoop || {};
    const statusTotals = stats.statusTotals || {};
    const enrichment = stats.enrichment || {};
    const parseCount = (value) => {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    };
    const truncate = (value, max = 120) => {
      if (!value) {
        return '';
      }
      const text = String(value);
      return text.length > max ? `${text.slice(0, max)}…` : text;
    };
    const items = [];

    const loopRuns = parseCount(loop.runs);
    const loopFound = parseCount(loop.discovered);
    if (loop.running) {
      const indicator = loop.stop_requested
        ? 'system-status__indicator--pending'
        : 'system-status__indicator--running';
      let text = `Discovery loop: Running (${formatNumber(loopRuns)} runs, ${formatNumber(loopFound)} new)`;
      if (loop.stop_requested) {
        text += ' — stop after current run';
      }
      items.push(this.renderSystemStatusItem(indicator, text));
    } else {
      let indicator = 'system-status__indicator--idle';
      let text = 'Discovery loop: Idle';
      const lastCompletedAt = formatDateTime(loop.last_completed_at);
      if (loop.last_reason === 'error') {
        indicator = 'system-status__indicator--error';
        text = `Discovery loop error after ${formatNumber(loopRuns)} runs`;
        if (loop.last_error) {
          text += ` — ${truncate(loop.last_error, 80)}`;
        }
      } else if (loop.last_reason === 'stopped' || loop.last_reason === 'completed') {
        const reasonLabel = loop.last_reason === 'stopped' ? 'stopped' : 'completed';
        text = `Discovery loop ${reasonLabel}: ${formatNumber(loopRuns)} runs, ${formatNumber(loopFound)} new`;
        if (lastCompletedAt) {
          text += ` (${lastCompletedAt})`;
        }
      } else if (lastCompletedAt) {
        text = `Discovery loop idle since ${lastCompletedAt}`;
      }
      items.push(this.renderSystemStatusItem(indicator, text));
    }

    const processingCount = parseCount(statusTotals.processing);
    const processingIndicator = processingCount > 0
      ? 'system-status__indicator--pending'
      : 'system-status__indicator--idle';
    const processingText = `Channels processing: ${formatNumber(processingCount)}`;
    items.push(this.renderSystemStatusItem(processingIndicator, processingText));

    const activeJobs = parseCount(enrichment.activeJobs);
    const pendingChannels = parseCount(enrichment.pendingChannels);
    let enrichmentIndicator = 'system-status__indicator--idle';
    let enrichmentText = 'Enrichment: Idle';
    if (activeJobs > 0) {
      enrichmentIndicator = 'system-status__indicator--running';
      const jobLabel = activeJobs === 1 ? 'job' : 'jobs';
      enrichmentText = `Enrichment: ${formatNumber(activeJobs)} ${jobLabel} active`;
      if (pendingChannels > 0) {
        enrichmentText += ` (${formatNumber(pendingChannels)} pending)`;
      }
    } else if (pendingChannels > 0 || parseCount(enrichment.processingChannels) > 0) {
      enrichmentIndicator = 'system-status__indicator--pending';
      const pendingTotal = pendingChannels || parseCount(enrichment.processingChannels);
      enrichmentText = `Enrichment: Finalizing ${formatNumber(pendingTotal)} channels`;
    }
    items.push(this.renderSystemStatusItem(enrichmentIndicator, enrichmentText));

    this.el.systemStatusBar.innerHTML = items.join('');
  }

  async maybeRefreshAfterLoop(stats) {
    if (!stats) {
      return;
    }
    const loop = stats.discoveryLoop || {};
    const statusTotals = stats.statusTotals || {};
    const enrichment = stats.enrichment || {};
    const processingCount = Number(statusTotals.processing ?? 0) || 0;
    const activeJobs = Number(enrichment.activeJobs ?? 0) || 0;
    if (loop.running) {
      return;
    }
    if (!loop.last_completed_at) {
      return;
    }
    if (processingCount > 0 || activeJobs > 0) {
      return;
    }
    const version = typeof loop.version === 'number' ? loop.version : null;
    if (version != null && version === this.lastLoopCompletionVersion) {
      return;
    }
    this.lastLoopCompletionVersion = version != null ? version : Date.now();
    await this.loadTable(Category.ACTIVE, (state) => ({ ...state, loading: true }));
    if (this.activeTab !== Category.ACTIVE) {
      await this.loadTable(this.activeTab, (state) => ({ ...state, loading: true }));
    }
    let message = 'Discovery loop finished. Data refreshed.';
    let tone = 'success';
    if (loop.last_reason === 'stopped') {
      message = 'Discovery loop stopped. Data refreshed.';
    } else if (loop.last_reason === 'completed') {
      message = 'Discovery loop completed. Data refreshed.';
    } else if (loop.last_reason === 'error') {
      message = 'Discovery loop ended with errors. Data refreshed.';
      tone = 'error';
    }
    this.updateStatusBar(message, tone);
  }

  updateStatusBar(message, tone = 'info') {
    if (!this.el.statusBar) {
      return;
    }
    clearTimeout(this.statusTimeout);
    if (!message) {
      this.el.statusBar.textContent = '';
      this.el.statusBar.hidden = true;
      this.el.statusBar.className = 'status-bar';
      return;
    }
    this.el.statusBar.textContent = message;
    this.el.statusBar.hidden = false;
    this.el.statusBar.className = `status-bar ${StatusToneClass[tone] || StatusToneClass.info}`;
    this.statusTimeout = setTimeout(() => {
      this.el.statusBar.hidden = true;
    }, 6000);
  }

  setProgress(message) {
    if (!this.el.progressBar) {
      return;
    }
    if (!message) {
      this.el.progressBar.hidden = true;
      this.el.progressBar.textContent = '';
    } else {
      this.el.progressBar.hidden = false;
      this.el.progressBar.textContent = message;
      this.el.progressBar.className = 'status-bar status-bar--info';
    }
  }

  setSummary(message, tone = 'info') {
    if (!this.el.summaryBar) {
      return;
    }
    if (!message) {
      this.el.summaryBar.hidden = true;
      this.el.summaryBar.textContent = '';
    } else {
      this.el.summaryBar.hidden = false;
      this.el.summaryBar.textContent = message;
      this.el.summaryBar.className = `status-bar ${StatusToneClass[tone] || StatusToneClass.info}`;
    }
  }

  async loadStats() {
    if (this.statsPromise) {
      return this.statsPromise;
    }
    this.statsPromise = (async () => {
      try {
        const stats = await fetchStats();
        this.stats = stats;
        this.updateQuickStats();
        this.updateSystemStatus(stats);
        await this.maybeRefreshAfterLoop(stats);
        return stats;
      } catch (error) {
        console.error('Failed to load stats', error);
        this.updateStatusBar('Failed to load statistics.', 'error');
        this.updateSystemStatus(null, { unavailable: true });
        return null;
      } finally {
        this.statsPromise = null;
      }
    })();
    return this.statsPromise;
  }

  async loadTable(category, mutate) {
    const table = this.tables[category];
    const snapshot = mutate ? mutate({ ...table, filters: { ...table.filters } }) : { ...table, filters: { ...table.filters } };
    snapshot.loading = true;
    this.tables[category] = snapshot;
    if (category === this.activeTab) {
      this.renderTable(true);
    }
    try {
      const { items, total } = await fetchChannels(
        category,
        snapshot.filters,
        snapshot.sort,
        snapshot.order,
        snapshot.limit,
        snapshot.page,
      );
      const rows = items;
      const effectiveTotal = total;
      const maxPage = effectiveTotal > 0 ? Math.max(0, Math.ceil(effectiveTotal / snapshot.limit) - 1) : 0;
      if (snapshot.page > maxPage) {
        const newPage = Math.max(0, maxPage);
        const nextState = { ...snapshot, page: newPage, loading: true };
        this.tables[category] = nextState;
        if (category === this.activeTab) {
          this.renderTable(true);
        }
        await this.loadTable(category, (state) => ({ ...state, page: newPage, loading: true }));
        return;
      }
      this.tables[category] = {
        ...snapshot,
        rows,
        total: effectiveTotal,
        loading: false,
        error: null,
      };
    } catch (error) {
      this.tables[category] = {
        ...snapshot,
        rows: [],
        loading: false,
        error: error instanceof Error ? error.message : 'Failed to load channels',
      };
      console.error('Failed to load channels', error);
    }
    if (category === this.activeTab) {
      this.renderTable();
      this.updatePagination();
    }
  }

  renderTable(loadingState = false) {
    const table = this.tables[this.activeTab];
    const info = CategoryInfo[this.activeTab];
    if (this.el.tableTitle) {
      this.el.tableTitle.textContent = `${info.icon} ${info.title}`;
    }
    if (this.el.tableSubtitle) {
      this.el.tableSubtitle.textContent = info.subtitle;
    }

    if (this.el.tableStatus) {
      if (loadingState) {
        this.el.tableStatus.textContent = 'Loading…';
      } else if (!table.rows.length) {
        this.el.tableStatus.textContent = 'No matching channels';
      } else {
        const startIndex = table.page * table.limit + 1;
        const endIndex = startIndex + table.rows.length - 1;
        this.el.tableStatus.textContent = `Showing ${startIndex} – ${endIndex} of ${formatNumber(table.total)}`;
      }
    }

    const primary = info.primaryAction;
    if (primary) {
      this.el.bulkPrimaryBtn.hidden = false;
      this.el.bulkPrimaryBtn.textContent = primary.text;
      this.el.bulkPrimaryBtn.className = `btn btn--${primary.tone}`;
      this.el.bulkPrimaryBtn.disabled = loadingState || this.enrichmentBusy;
    } else {
      this.el.bulkPrimaryBtn.hidden = true;
    }
    const secondary = info.secondaryAction;
    if (secondary) {
      this.el.bulkSecondaryBtn.hidden = false;
      this.el.bulkSecondaryBtn.textContent = secondary.text;
      this.el.bulkSecondaryBtn.className = `btn btn--${secondary.tone}`;
      this.el.bulkSecondaryBtn.disabled = loadingState || this.enrichmentBusy;
    } else {
      this.el.bulkSecondaryBtn.hidden = true;
    }
    if (this.el.exportCsvBtn) {
      this.el.exportCsvBtn.disabled = loadingState;
    }
    if (this.el.exportBundleBtn) {
      this.el.exportBundleBtn.disabled = loadingState;
    }

    this.el.tabButtons.forEach((button) => {
      button.classList.toggle('is-active', button.dataset.category === this.activeTab);
    });

    if (loadingState) {
      this.el.tableBody.innerHTML = '<tr><td colspan="9" class="table-loading">Loading channels…</td></tr>';
      return;
    }

    if (table.error) {
      if (this.el.tableStatus) {
        this.el.tableStatus.textContent = 'Failed to load channels';
      }
      this.el.tableBody.innerHTML = `<tr><td colspan="9" class="table-error">${table.error}</td></tr>`;
      return;
    }

    if (!table.rows.length) {
    this.el.tableBody.innerHTML = '<tr><td colspan="9" class="table-empty">No channels match the current filters.</td></tr>';
      return;
    }

    const rowsHtml = table.rows
      .map((row) => this.renderRow(row))
      .join('');
    this.el.tableBody.innerHTML = rowsHtml;
    this.updatePagination();
  }

  renderRow(row) {
    const category = this.activeTab;
    const badgeClass = StatusBadgeClass[category] || 'status-chip';
    const badgeIcon = StatusBadgeIcon[category] || '';
    const statusText = row.status ? row.status.toUpperCase() : 'UNKNOWN';
    const statusReason = row.status_reason || row.last_error || '';
    const actions = [];
    if (category === Category.ACTIVE) {
      actions.push({ action: 'archive', label: '🟡 Archive', tone: 'yellow' });
      actions.push({ action: 'blacklist', label: '🔴 Blacklist', tone: 'red' });
    } else if (category === Category.ARCHIVED) {
      actions.push({ action: 'restore', label: '🟢 Restore', tone: 'green' });
      actions.push({ action: 'blacklist', label: '🔴 Blacklist', tone: 'red' });
    } else if (category === Category.BLACKLISTED) {
      actions.push({ action: 'restore', label: '🟢 Restore', tone: 'green' });
    }

    const safeIdAttr = escapeHtml(row.channel_id);
    const actionButtons = actions
      .map(
        (action) =>
          `<button class="btn btn--${action.tone}" data-action="${action.action}" data-id="${safeIdAttr}">${action.label}</button>`
      )
      .join('');

    const language = row.language ? escapeHtml(row.language.toUpperCase()) : '—';
    const subscribers = row.subscribers != null ? formatNumber(row.subscribers) : '—';
    const rawUrl = row.url && /^https?:/i.test(row.url)
      ? row.url
      : `https://www.youtube.com/channel/${row.channel_id}`;
    const url = escapeHtml(rawUrl);
    const name = escapeHtml(row.name || 'Unnamed channel');
    const channelId = escapeHtml(row.channel_id);
    const emails = row.emails ? escapeHtml(row.emails) : '—';
    const reason = statusReason ? escapeHtml(statusReason) : '';
    const emailGate = Boolean(row.email_gate_present);
    const duplicateCount = Number(row.duplicate_email_count || 0);
    const badges = [];
    if (emailGate) {
      badges.push('<span class="badge badge--gate">🔒 Email gate</span>');
    }
    if (duplicateCount > 0) {
      const duplicateLabel = duplicateCount > 1 ? 'Duplicate emails' : 'Duplicate email';
      const duplicateTitle = row.duplicate_emails
        ? `Also seen on other channels: ${row.duplicate_emails}`
        : 'Email address also appears on other channels';
      badges.push(
        `<span class="badge badge--duplicate" title="${escapeHtml(duplicateTitle)}">⚠️ ${duplicateLabel}</span>`
      );
    }
    const badgesHtml = badges.length ? `<div class="name-badges">${badges.join('')}</div>` : '';

    return `
      <tr>
        <td class="name-cell">
          <span class="name-primary">${name}</span>
          <a class="name-link" href="${url}" target="_blank" rel="noopener noreferrer">${channelId}</a>
          ${badgesHtml}
        </td>
        <td>${subscribers}</td>
        <td>${language}</td>
        <td class="emails-cell">${emails}</td>
        <td>
          <span class="status-chip ${badgeClass}">${badgeIcon} ${escapeHtml(statusText)}</span>
          ${reason ? `<span class="status-reason">${reason}</span>` : ''}
        </td>
        <td>${formatDate(row.last_updated || row.created_at)}</td>
        <td>${formatDate(row.exported_at)}</td>
        <td>${formatDate(row.archived_at)}</td>
        <td class="actions-cell">${actionButtons}</td>
      </tr>
    `;
  }

  updatePagination() {
    const table = this.tables[this.activeTab];
    const totalPages = Math.max(1, Math.ceil(table.total / table.limit));
    const currentPage = Math.min(table.page + 1, totalPages);
    this.el.paginationLabel.textContent = `Page ${currentPage} of ${totalPages} • Showing ${table.rows.length} / ${formatNumber(table.total)} results`;
    this.el.prevPageBtn.disabled = table.page === 0;
    this.el.nextPageBtn.disabled = table.page + 1 >= totalPages;
  }

  applyFiltersFromUI() {
    const filters = {
      query: this.el.filterQuery.value.trim(),
      languages: parseList(this.el.filterLanguages.value),
      minSubscribers: this.el.filterMinSubs.value ? Number(this.el.filterMinSubs.value) : '',
      maxSubscribers: this.el.filterMaxSubs.value ? Number(this.el.filterMaxSubs.value) : '',
      emailsOnly: this.el.filterEmailsOnly.checked,
      uniqueEmails: this.el.filterUniqueEmails.checked,
      emailGateOnly: this.el.filterEmailGateOnly.checked,
      statuses: this.el.filterStatusCheckboxes.filter((c) => c.checked).map((c) => c.value),
    };
    this.tables[this.activeTab].filters = filters;
    this.tables[this.activeTab].page = 0;
    this.loadTable(this.activeTab, (state) => ({
      ...state,
      filters,
      page: 0,
      loading: true,
    }));
  }

  changeTab(category) {
    this.activeTab = category;
    this.renderTabs();
    this.applyFiltersToUI(this.tables[category].filters);
    this.updateSortInputs();
    this.renderTable();
    this.updatePagination();
    if (!this.tables[category].rows.length && !this.tables[category].loading) {
      this.loadTable(category, (state) => ({ ...state, loading: true }));
    }
  }

  updateSort(sort) {
    const table = this.tables[this.activeTab];
    table.sort = sort;
    table.page = 0;
    this.loadTable(this.activeTab, (state) => ({ ...state, sort, page: 0, loading: true }));
  }

  updateOrder(order) {
    const table = this.tables[this.activeTab];
    table.order = order;
    table.page = 0;
    this.loadTable(this.activeTab, (state) => ({ ...state, order, page: 0, loading: true }));
  }

  changePage(delta) {
    const table = this.tables[this.activeTab];
    const nextPage = table.page + delta;
    if (nextPage < 0) {
      return;
    }
    const totalPages = Math.ceil(table.total / table.limit);
    if (totalPages && nextPage >= totalPages) {
      return;
    }
    table.page = nextPage;
    this.loadTable(this.activeTab, (state) => ({ ...state, page: nextPage, loading: true }));
  }

  async handleRowAction(action, channelId) {
    try {
      if (action === 'archive') {
        await archiveChannels([channelId]);
        this.updateStatusBar('Channel archived.', 'success');
        await this.afterMove(Category.ACTIVE, Category.ARCHIVED);
      } else if (action === 'blacklist') {
        await blacklistChannels([channelId], this.activeTab);
        this.updateStatusBar('Channel blacklisted.', 'success');
        await this.afterMove(this.activeTab, Category.BLACKLISTED);
      } else if (action === 'restore') {
        await restoreChannels([channelId], this.activeTab);
        this.updateStatusBar('Channel restored to Active.', 'success');
        await this.afterMove(this.activeTab, Category.ACTIVE);
      }
    } catch (error) {
      console.error('Row action failed', error);
      this.updateStatusBar(error instanceof Error ? error.message : 'Action failed.', 'error');
    }
  }

  async handleBulkPrimary() {
    const info = CategoryInfo[this.activeTab];
    if (!info || !info.primaryAction) {
      return;
    }
    try {
      if (this.activeTab === Category.ACTIVE) {
        const rows = this.tables[this.activeTab].rows || [];
        const ids = rows.map((row) => row.channel_id).filter(Boolean);
        if (!ids.length) {
          this.updateStatusBar('No channels on this page to archive.', 'info');
          return;
        }
        const result = await archiveChannels(ids);
        const archivedCount = Number(result?.archived ?? 0);
        if (archivedCount > 0) {
          const label = archivedCount === 1 ? 'channel' : 'channels';
          this.updateStatusBar(`Archived ${formatNumber(archivedCount)} ${label} from the current filter.`, 'success');
          await this.afterMove(Category.ACTIVE, Category.ARCHIVED);
        } else {
          this.updateStatusBar('No channels were archived.', 'info');
        }
      } else if (this.activeTab === Category.ARCHIVED) {
        await restoreByFilter(
          this.activeTab,
          this.tables[this.activeTab].filters,
          this.tables[this.activeTab].sort,
          this.tables[this.activeTab].order,
          this.tables[this.activeTab].limit,
          this.tables[this.activeTab].page,
        );
        this.updateStatusBar('Filtered channels restored.', 'success');
        await this.afterMove(Category.ARCHIVED, Category.ACTIVE);
      } else if (this.activeTab === Category.BLACKLISTED) {
        await restoreByFilter(
          this.activeTab,
          this.tables[this.activeTab].filters,
          this.tables[this.activeTab].sort,
          this.tables[this.activeTab].order,
          this.tables[this.activeTab].limit,
          this.tables[this.activeTab].page,
        );
        this.updateStatusBar('Blacklisted channels restored to Active.', 'success');
        await this.afterMove(Category.BLACKLISTED, Category.ACTIVE);
      }
    } catch (error) {
      console.error('Bulk primary action failed', error);
      this.updateStatusBar(error instanceof Error ? error.message : 'Bulk action failed.', 'error');
    }
  }

  async handleBulkSecondary() {
    const info = CategoryInfo[this.activeTab];
    if (!info || !info.secondaryAction) {
      return;
    }
    try {
      if (this.activeTab === Category.ACTIVE) {
        await blacklistByFilter(
          this.activeTab,
          this.tables[this.activeTab].filters,
          this.tables[this.activeTab].sort,
          this.tables[this.activeTab].order,
          this.tables[this.activeTab].limit,
          this.tables[this.activeTab].page,
        );
        this.updateStatusBar('Filtered channels blacklisted.', 'success');
        await this.afterMove(Category.ACTIVE, Category.BLACKLISTED);
      } else if (this.activeTab === Category.ARCHIVED) {
        await blacklistByFilter(
          this.activeTab,
          this.tables[this.activeTab].filters,
          this.tables[this.activeTab].sort,
          this.tables[this.activeTab].order,
          this.tables[this.activeTab].limit,
          this.tables[this.activeTab].page,
        );
        this.updateStatusBar('Archived channels blacklisted.', 'success');
        await this.afterMove(Category.ARCHIVED, Category.BLACKLISTED);
      }
    } catch (error) {
      console.error('Bulk secondary action failed', error);
      this.updateStatusBar(error instanceof Error ? error.message : 'Bulk action failed.', 'error');
    }
  }

  async afterMove(source, destination) {
    await this.loadStats();
    await this.loadTable(source, (state) => ({ ...state, loading: true }));
    if (destination && destination !== source) {
      await this.loadTable(destination, (state) => ({ ...state, loading: true }));
    }
  }

  async handleExportCsv() {
    try {
      const table = this.tables[this.activeTab];
      const { csv, exportTimestamp } = await downloadCsv(
        this.activeTab,
        table.filters,
        table.sort,
        table.order,
      );
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      link.href = url;
      link.download = `${this.activeTab}-channels-${timestamp}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      this.updateStatusBar('CSV exported successfully.', 'success');
      if (this.activeTab === Category.ACTIVE && exportTimestamp && table.total > 0) {
        const confirmArchive = window.confirm('Archive exported rows?');
        if (confirmArchive) {
          await this.handleArchiveExportedRows(exportTimestamp);
        }
      }
    } catch (error) {
      console.error('CSV export failed', error);
      this.updateStatusBar(error instanceof Error ? error.message : 'CSV export failed.', 'error');
    }
  }

  async handleExportBundle() {
    try {
      this.setProgress('Preparing project backup…');
      const { blob, exportTimestamp } = await downloadBundle();
      const url = URL.createObjectURL(blob);
      const timestampSource = exportTimestamp || new Date().toISOString();
      const timestamp = timestampSource.replace(/[:.]/g, '-');
      const link = document.createElement('a');
      link.href = url;
      link.download = `project-bundle-${timestamp}.zip`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      this.updateStatusBar('Project bundle exported successfully.', 'success');
    } catch (error) {
      console.error('Project bundle export failed', error);
      this.updateStatusBar(
        error instanceof Error ? error.message : 'Project bundle export failed.',
        'error',
      );
    } finally {
      this.setProgress('');
    }
  }

  async handleArchiveExportedRows(exportTimestamp) {
    try {
      this.setProgress('Archiving exported rows…');
      const result = await archiveExportedRows(exportTimestamp);
      const archivedCount = Number(result?.archived ?? 0);
      if (archivedCount > 0) {
        const label = archivedCount === 1 ? 'channel' : 'channels';
        this.updateStatusBar(`Archived ${formatNumber(archivedCount)} exported ${label}.`, 'success');
        await this.afterMove(Category.ACTIVE, Category.ARCHIVED);
      } else {
        this.updateStatusBar('No exported rows were archived.', 'info');
      }
    } catch (error) {
      console.error('Archiving exported rows failed', error);
      this.updateStatusBar(
        error instanceof Error ? error.message : 'Failed to archive exported rows.',
        'error',
      );
    } finally {
      this.setProgress('');
    }
  }

  async performDiscoveryRun(inputs, options = {}) {
    const { statusLabel, deferStatusUpdate = false } = options;
    const prefix = statusLabel ? `${statusLabel}: ` : '';
    this.updateStatusBar(`${prefix}Starting discovery…`, 'info');
    try {
      const requestOptions = {};
      if (Array.isArray(inputs.denyLanguages) && inputs.denyLanguages.length) {
        requestOptions.denyLanguages = inputs.denyLanguages;
      }
      if (inputs.lastUploadMaxAgeDays != null) {
        requestOptions.lastUploadMaxAgeDays = inputs.lastUploadMaxAgeDays;
      }
      const response = await discoverChannels(inputs.keywords, inputs.perKeyword, requestOptions);
      let message = `Discovered ${response.found} new channels.`;
      if (response.blacklisted) {
        const suffix = response.blacklisted === 1 ? 'candidate' : 'candidates';
        message += ` Blacklisted ${response.blacklisted} ${suffix}.`;
      }
      if (!deferStatusUpdate) {
        this.updateStatusBar(`${prefix}${message}`, 'success');
      }
      await this.loadStats();
      await this.loadTable(Category.ACTIVE, (state) => ({ ...state, loading: true }));
      await this.triggerAutoEnrich(response.found);
      return { response, message };
    } catch (error) {
      console.error('Discovery failed', error);
      const errorMessage = error instanceof Error ? error.message : 'Discovery failed.';
      this.updateStatusBar(`${prefix}${errorMessage}`, 'error');
      return null;
    }
  }

  async handleEnrichmentSettingsSubmit() {
    const scope = this.el.enrichScopeAll?.checked ? 'all_active' : 'filtered';
    const emailMode = this.el.enrichEmailsEnabled?.checked
      ? this.el.enrichEmailsMode?.value || DEFAULT_EMAIL_MODE
      : 'off';
    const languageMode = this.el.enrichLanguageEnabled?.checked
      ? this.el.enrichLanguageMode?.value || DEFAULT_LANGUAGE_MODE
      : 'off';
    const options = {
      ...emailOptionsFromMode(emailMode),
      ...languageOptionsFromMode(languageMode),
      update_metadata: Boolean(this.el.enrichUpdateMetadata?.checked),
      update_activity: Boolean(this.el.enrichUpdateActivity?.checked),
    };
    this.enrichmentSettings = this.normalizeEnrichmentSettings({
      scope,
      options,
      emailMode,
      languageMode,
    });
    this.saveEnrichmentSettings();
    this.updateEnrichmentLastUsed();
    this.closeEnrichmentSettings();
    await this.handleEnrich(options, { scope, emailMode, languageMode });
  }

  async handleEnrich(options, overrides = {}) {
    if (this.enrichmentBusy) {
      return;
    }
    const { limitOverride = null, autoTriggered = false, scope: scopeOverride = null } = overrides;
    let limitValue = null;
    if (limitOverride != null) {
      limitValue = Number(limitOverride);
    } else if (this.discoverySettings?.enrichLimit != null) {
      limitValue = Number(this.discoverySettings.enrichLimit);
    }
    if (limitValue != null && (Number.isNaN(limitValue) || limitValue <= 0)) {
      this.updateStatusBar('Enrichment limit must be a positive number.', 'error');
      return;
    }

    const appliedOptions = normalizeEnrichmentOptions(
      options || this.enrichmentSettings?.options || defaultEnrichmentOptions(),
    );
    const emailMode =
      overrides.emailMode ||
      this.enrichmentSettings?.emailMode ||
      deriveEmailModeFromOptions(appliedOptions);
    const languageMode =
      overrides.languageMode ||
      this.enrichmentSettings?.languageMode ||
      deriveLanguageModeFromOptions(appliedOptions);
    const scope = scopeOverride || this.enrichmentSettings?.scope || 'filtered';

    try {
      this.enrichmentBusy = true;
      this.autoEnrichQueuedNotified = false;
      this.el.enrichBtn.disabled = true;
      this.renderTable();
      const request = {
        limit: limitValue,
        options: appliedOptions,
        emails_mode: emailMode,
        language_mode: languageMode,
        scope,
        category: this.activeTab,
        forceRun: this.el.enrichForceToggle?.checked ?? false,
        neverReenrich: this.el.enrichNeverToggle?.checked ?? false,
      };
      if (scope === 'filtered') {
        request.filters = this.tables[this.activeTab]?.filters;
      }

      const { jobId, total, skipped = 0 } = await startEnrichment(request);
      let message;
      if (autoTriggered) {
        if (limitValue != null) {
          const label = Math.abs(limitValue) === 1 ? 'channel' : 'channels';
          message = `Auto-enrich job ${jobId} started for ${formatNumber(limitValue)} ${label}.`;
        } else {
          message = `Auto-enrich job ${jobId} started.`;
        }
      } else {
        message = `Enrichment job ${jobId} started.`;
      }
      if (skipped > 0) {
        const label = skipped === 1 ? 'channel' : 'channels';
        message += ` Skipped ${skipped} ${label} due to recent no-email results.`;
      }
      this.updateStatusBar(message, 'success');
      this.setSummary('', 'info');
      this.openEventSource(jobId, total);
    } catch (error) {
      this.enrichmentBusy = false;
      this.el.enrichBtn.disabled = false;
      this.renderTable();
      if (autoTriggered && limitValue != null && limitValue > 0) {
        this.pendingAutoEnrich += Number(limitValue);
        this.autoEnrichQueuedNotified = false;
      }
      console.error('Enrichment failed', error);
      this.updateStatusBar(error instanceof Error ? error.message : 'Failed to start enrichment.', 'error');
    }
  }

  isAutoEnrichEnabled() {
    return Boolean(this.discoverySettings?.autoEnrichEnabled);
  }

  getAutoEnrichOptions() {
    const settings = this.discoverySettings || defaultDiscoverySettingsState();
    if (settings.autoEnrichOptions) {
      return normalizeEnrichmentOptions(settings.autoEnrichOptions);
    }
    return settings.autoEnrichMode === 'full'
      ? defaultEnrichmentOptions()
      : emailOnlyEnrichmentOptions();
  }

  async triggerAutoEnrich(newChannelsCount) {
    const count = Number(newChannelsCount) || 0;
    if (count <= 0) {
      return;
    }
    if (!this.isAutoEnrichEnabled()) {
      this.pendingAutoEnrich = 0;
      return;
    }
    if (this.enrichmentBusy) {
      this.pendingAutoEnrich += count;
      if (!this.autoEnrichQueuedNotified) {
        this.updateStatusBar('Auto-enrich queued until the current enrichment finishes.', 'info');
        this.autoEnrichQueuedNotified = true;
      }
      return;
    }
    const options = this.getAutoEnrichOptions();
    await this.handleEnrich(options, {
      limitOverride: count,
      autoTriggered: true,
      scope: 'all_active',
    });
  }

  async processPendingAutoEnrich() {
    if (!this.pendingAutoEnrich || this.pendingAutoEnrich <= 0) {
      this.pendingAutoEnrich = 0;
      return;
    }
    if (!this.isAutoEnrichEnabled()) {
      this.pendingAutoEnrich = 0;
      return;
    }
    const queued = this.pendingAutoEnrich;
    this.pendingAutoEnrich = 0;
    this.autoEnrichQueuedNotified = false;
    await this.triggerAutoEnrich(queued);
  }

  async reportDiscoveryLoopProgress() {
    try {
      await notifyDiscoveryLoopProgress({
        runs: this.discoveryLoopStats.runs,
        discovered: this.discoveryLoopStats.found,
      });
    } catch (error) {
      console.warn('Failed to sync discovery loop progress', error);
    }
  }

  queueDiscoveryLoopCompletion(payload) {
    this.discoveryLoopCompletionPayload = {
      runs: Number(payload.runs ?? 0) || 0,
      discovered: Number(payload.discovered ?? 0) || 0,
      reason: payload.reason || null,
      error: Boolean(payload.error),
      message: payload.message ? String(payload.message) : undefined,
    };
    this.discoveryLoopCompletionSent = false;
    this.tryFinalizeDiscoveryLoop();
  }

  async tryFinalizeDiscoveryLoop() {
    if (!this.discoveryLoopCompletionPayload) {
      return;
    }
    if (this.discoveryLoopActive || this.enrichmentBusy || this.pendingAutoEnrich > 0) {
      return;
    }
    if (this.finalizingDiscoveryLoop) {
      return;
    }
    const payload = { ...this.discoveryLoopCompletionPayload };
    this.finalizingDiscoveryLoop = true;
    try {
      const requestPayload = {
        runs: Number(payload.runs ?? 0) || 0,
        discovered: Number(payload.discovered ?? 0) || 0,
      };
      if (payload.reason) {
        requestPayload.reason = payload.reason;
      }
      if (payload.error) {
        requestPayload.error = true;
      }
      if (payload.message) {
        requestPayload.message = payload.message;
      }
      await notifyDiscoveryLoopComplete(requestPayload);
      this.discoveryLoopCompletionPayload = null;
      this.discoveryLoopCompletionSent = true;
    } catch (error) {
      console.warn('Failed to notify discovery loop completion', error);
      setTimeout(() => this.tryFinalizeDiscoveryLoop(), 3000);
    } finally {
      this.finalizingDiscoveryLoop = false;
    }
  }

  applyChannelUpdate(update) {
    if (!update || !update.channelId) {
      return;
    }
    const channelId = update.channelId;
    const categories = [Category.ACTIVE, Category.ARCHIVED, Category.BLACKLISTED];
    let shouldRender = false;
    const mapEmails = (value) => {
      if (!value) {
        return null;
      }
      if (Array.isArray(value)) {
        if (!value.length) {
          return null;
        }
        return value.join(', ');
      }
      return String(value);
    };
    categories.forEach((category) => {
      const table = this.tables[category];
      if (!table || !Array.isArray(table.rows) || !table.rows.length) {
        return;
      }
      const index = table.rows.findIndex((row) => row.channel_id === channelId);
      if (index === -1) {
        return;
      }
      const current = table.rows[index];
      const nextRow = { ...current };
      if (update.status) {
        nextRow.status = update.status;
      }
      if ('statusReason' in update) {
        nextRow.status_reason = update.statusReason || null;
      }
      if ('lastStatusChange' in update) {
        nextRow.last_status_change = update.lastStatusChange || null;
      }
      if ('emails' in update) {
        const mapped = mapEmails(update.emails);
        nextRow.emails = mapped;
      }
      if ('subscribers' in update && update.subscribers != null) {
        nextRow.subscribers = update.subscribers;
      }
      if ('language' in update && update.language != null) {
        nextRow.language = update.language;
      }
      if ('languageConfidence' in update && update.languageConfidence != null) {
        nextRow.language_confidence = update.languageConfidence;
      }
      if ('lastUpdated' in update && update.lastUpdated) {
        nextRow.last_updated = update.lastUpdated;
      }
      if ('emailGatePresent' in update) {
        nextRow.email_gate_present = update.emailGatePresent;
      }
      const nextRows = [...table.rows];
      nextRows[index] = nextRow;
      this.tables[category] = { ...table, rows: nextRows };
      if (category === this.activeTab) {
        shouldRender = true;
      }
    });
    if (shouldRender) {
      this.renderTable();
    }
  }

  async startDiscoveryLoop(initialInputs = null) {
    if (this.discoveryLoopActive) {
      this.updateStatusBar('Discovery loop is already running.', 'info');
      return;
    }
    const inputs = initialInputs ?? this.collectDiscoveryInputsFromSettings();
    if (!inputs) {
      return;
    }
    this.discoveryLoopStats = { runs: 0, found: 0 };
    this.discoveryLoopStopping = false;
    this.discoveryLoopCompletionPayload = null;
    this.discoveryLoopCompletionSent = false;
    this.finalizingDiscoveryLoop = false;
    this.setDiscoveryLoopRunningState(true);
    this.updateDiscoveryLoopCounter();
    this.updateStatusBar('Discovery loop started. Click stop to finish.', 'info');
    try {
      await notifyDiscoveryLoopStart({ runs: 0, discovered: 0 });
    } catch (error) {
      console.warn('Failed to notify discovery loop start', error);
    }

    let loopError = false;
    try {
      while (!this.discoveryLoopStopping) {
        const runIndex = this.discoveryLoopStats.runs + 1;
        const label = `Run #${runIndex}`;
        const result = await this.performDiscoveryRun(inputs, {
          statusLabel: label,
          deferStatusUpdate: true,
        });
        if (!result) {
          loopError = true;
          break;
        }
        const { response, message } = result;
        this.discoveryLoopStats.runs += 1;
        this.discoveryLoopStats.found += Number(response?.found ?? 0);
        this.updateDiscoveryLoopCounter();
        await this.reportDiscoveryLoopProgress();
        const totalMessage = `${label}: ${message} Total discovered: ${formatNumber(
          this.discoveryLoopStats.found,
        )}.`;
        this.updateStatusBar(totalMessage, 'success');
        if (this.discoveryLoopStopping) {
          break;
        }
        await this.delay(2000);
        if (this.discoveryLoopStopping) {
          break;
        }
      }
    } finally {
      const stopRequested = this.discoveryLoopStopping;
      const finalRuns = this.discoveryLoopStats.runs;
      const finalFound = this.discoveryLoopStats.found;
      this.discoveryLoopStopping = false;
      this.setDiscoveryLoopRunningState(false);
      this.updateDiscoveryLoopCounter();
      if (!loopError) {
        if (finalRuns > 0) {
          const runLabel = finalRuns === 1 ? 'run' : 'runs';
          const channelLabel = finalFound === 1 ? 'channel' : 'channels';
          this.updateStatusBar(
            `Discovery loop stopped after ${formatNumber(finalRuns)} ${runLabel} with ${formatNumber(finalFound)} new ${channelLabel}.`,
            'info',
          );
        } else {
          this.updateStatusBar('Discovery loop stopped.', 'info');
        }
      }
      const completionReason = loopError
        ? 'error'
        : stopRequested
        ? 'stopped'
        : 'completed';
      this.queueDiscoveryLoopCompletion({
        runs: finalRuns,
        discovered: finalFound,
        reason: completionReason,
        error: loopError,
      });
    }
  }

  async stopDiscoveryLoop() {
    if (!this.discoveryLoopActive || this.discoveryLoopStopping) {
      return;
    }
    this.discoveryLoopStopping = true;
    if (this.el.discoverStopBtn) {
      this.el.discoverStopBtn.disabled = true;
    }
    this.updateStatusBar('Stopping discovery after current run…', 'info');
    try {
      await notifyDiscoveryLoopStop();
    } catch (error) {
      console.warn('Failed to notify discovery loop stop', error);
    }
  }

  setDiscoveryLoopRunningState(running) {
    this.discoveryLoopActive = running;
    if (this.el.discoverBtn) {
      this.el.discoverBtn.disabled = running;
    }
    if (this.el.discoverStopBtn) {
      if (running) {
        this.el.discoverStopBtn.removeAttribute('hidden');
        this.el.discoverStopBtn.disabled = false;
      } else {
        this.el.discoverStopBtn.setAttribute('hidden', 'true');
        this.el.discoverStopBtn.disabled = false;
      }
    }
    if (this.el.discoverRunCounter) {
      if (running) {
        this.el.discoverRunCounter.removeAttribute('hidden');
      } else {
        this.el.discoverRunCounter.setAttribute('hidden', 'true');
      }
    }
  }

  updateDiscoveryLoopCounter() {
    if (!this.el.discoverRunCounter) {
      return;
    }
    const runs = this.discoveryLoopStats.runs;
    const found = this.discoveryLoopStats.found;
    const runLabel = runs === 1 ? 'Run' : 'Runs';
    const channelLabel = found === 1 ? 'channel' : 'channels';
    this.el.discoverRunCounter.textContent = `${runLabel}: ${formatNumber(runs)} • New ${channelLabel}: ${formatNumber(found)}`;
  }

  delay(ms) {
    return new Promise((resolve) => {
      setTimeout(resolve, ms);
    });
  }

  toggleDropdown(toggleEl, menuEl) {
    if (!toggleEl || !menuEl || toggleEl.disabled) {
      return;
    }
    if (this.activeDropdown === menuEl) {
      this.closeDropdown(menuEl, toggleEl);
      return;
    }
    this.closeAllDropdowns();
    menuEl.removeAttribute('hidden');
    toggleEl.setAttribute('aria-expanded', 'true');
    this.activeDropdown = menuEl;
    this.activeDropdownToggle = toggleEl;
  }

  closeDropdown(menuEl, toggleEl = null) {
    if (!menuEl) {
      return;
    }
    menuEl.setAttribute('hidden', 'true');
    const toggle = toggleEl || this.activeDropdownToggle;
    if (toggle) {
      toggle.setAttribute('aria-expanded', 'false');
    }
    if (this.activeDropdown === menuEl) {
      this.activeDropdown = null;
      this.activeDropdownToggle = null;
    }
  }

  closeAllDropdowns() {
    if (this.activeDropdown) {
      this.closeDropdown(this.activeDropdown, this.activeDropdownToggle);
    }
  }

  openHelpModal() {
    if (!this.el.helpModal) {
      return;
    }
    this.closeAllDropdowns();
    this.el.helpModal.hidden = false;
    this.el.helpModal.setAttribute('aria-hidden', 'false');
  }

  closeHelpModal() {
    if (!this.el.helpModal) {
      return;
    }
    this.el.helpModal.hidden = true;
    this.el.helpModal.setAttribute('aria-hidden', 'true');
  }

  handleDocumentClick(event) {
    if (!this.activeDropdown) {
      return;
    }
    const target = event.target;
    if (typeof Node === 'undefined' || !(target instanceof Node)) {
      this.closeAllDropdowns();
      return;
    }
    if (
      this.activeDropdown.contains(target) ||
      (this.activeDropdownToggle && this.activeDropdownToggle.contains(target))
    ) {
      return;
    }
    this.closeAllDropdowns();
  }

  handleDocumentKeydown(event) {
    if (event.key === 'Escape') {
      this.closeAllDropdowns();
      this.closeHelpModal();
    }
  }

  openEventSource(jobId, total) {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
    this.setProgress(`Job ${jobId} queued…`);
    this.eventSource = new EventSource(`/api/enrich/stream/${jobId}`);
    this.eventSource.onmessage = async (event) => {
      if (!event.data) {
        return;
      }
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'channel') {
          this.applyChannelUpdate(payload);
        } else if (payload.type === 'progress') {
          const completed = payload.completed ?? 0;
          const pending = payload.pending ?? Math.max(0, (payload.total ?? total ?? 0) - completed);
          this.setProgress(
            `Job ${payload.jobId}: ${completed}/${payload.total ?? total ?? '?'} complete • ${pending} pending • ${payload.errors ?? 0} errors`
          );
          if (payload.done) {
            this.eventSource?.close();
            this.eventSource = null;
            this.enrichmentBusy = false;
            this.el.enrichBtn.disabled = false;
            this.setProgress('');
            const skipped = payload.skipped ?? 0;
            const requested = payload.requested;
            let summary = 'Enrichment job completed.';
            if (typeof requested === 'number') {
              summary += ` Processed ${completed} of ${requested} channels.`;
            }
            if (skipped > 0) {
              const label = skipped === 1 ? 'channel' : 'channels';
              summary += ` Skipped ${skipped} ${label} due to recent no-email results.`;
            }
            this.setSummary(summary, 'success');
            if (skipped > 0) {
              const label = skipped === 1 ? 'channel' : 'channels';
              this.updateStatusBar(
                `Skipped ${skipped} ${label} because they were recently enriched without emails.`,
                'info'
              );
            }
            await this.loadStats();
            await this.loadTable(Category.ACTIVE, (state) => ({ ...state, loading: true }));
            if (this.activeTab !== Category.ACTIVE) {
              await this.loadTable(this.activeTab, (state) => ({ ...state, loading: true }));
            }
            this.autoEnrichQueuedNotified = false;
            await this.processPendingAutoEnrich();
            this.tryFinalizeDiscoveryLoop();
          }
        } else if (payload.type === 'error') {
          this.updateStatusBar(payload.message || 'Enrichment error encountered.', 'error');
        }
      } catch (err) {
        console.error('Failed to parse SSE payload', err);
      }
    };
    this.eventSource.onerror = (event) => {
      console.error('EventSource error', event);
      this.setProgress('');
      this.updateStatusBar('Connection lost to enrichment stream.', 'error');
      this.eventSource?.close();
      this.eventSource = null;
      this.enrichmentBusy = false;
      this.el.enrichBtn.disabled = false;
      this.renderTable();
    };
  }

  openModal() {
    this.closeAllDropdowns();
    this.el.modal.removeAttribute('hidden');
    this.el.modalSummary.innerHTML = '';
    this.el.modalSummary.className = 'modal-summary';
    this.el.modalForm.reset();
    if (this.el.modalFile) {
      this.el.modalFile.value = '';
      requestAnimationFrame(() => {
        try {
          if (typeof this.el.modalFile.showPicker === 'function') {
            this.el.modalFile.showPicker();
          } else {
            this.el.modalFile.click();
          }
        } catch (err) {
          console.warn('File picker showPicker failed, falling back to click()', err);
          this.el.modalFile.click();
        }
      });
    }
  }

  closeModal() {
    this.el.modal.setAttribute('hidden', 'true');
  }

  renderBlacklistImportResult(result) {
    const summaryEl = this.el.modalSummary;
    if (!summaryEl) {
      return;
    }
    if (!result || typeof result !== 'object') {
      summaryEl.textContent = 'Import completed.';
      summaryEl.className = 'modal-summary modal-summary--success';
      return;
    }

    const countsSource = result.counts || {};
    const counts = {
      created:
        typeof countsSource.created === 'number'
          ? countsSource.created
          : Array.isArray(result.created)
            ? result.created.length
            : 0,
      updated:
        typeof countsSource.updated === 'number'
          ? countsSource.updated
          : Array.isArray(result.updated)
            ? result.updated.length
            : 0,
      skipped:
        typeof countsSource.skipped === 'number'
          ? countsSource.skipped
          : Array.isArray(result.skipped)
            ? result.skipped.length
            : 0,
      unresolved:
        typeof countsSource.unresolved === 'number'
          ? countsSource.unresolved
          : Array.isArray(result.unresolved)
            ? result.unresolved.length
            : 0,
      processed:
        typeof countsSource.processed === 'number' ? countsSource.processed : null,
    };

    const summaryParts = [
      `${counts.created} created`,
      `${counts.updated} updated`,
      `${counts.skipped} skipped`,
      `${counts.unresolved} unresolved`,
    ];
    const summaryLine = `Imported: ${summaryParts.join(' • ')}`;
    let html = `<p class="modal-summary__headline">${escapeHtml(summaryLine)}</p>`;

    if (counts.processed != null) {
      const processedText = `Processed ${counts.processed} row${counts.processed === 1 ? '' : 's'}.`;
      html += `<p class="modal-summary__meta-line">${escapeHtml(processedText)}</p>`;
    }

    if (counts.skipped > 0) {
      html +=
        '<p class="modal-summary__meta-line">Skipped entries include duplicates or already blacklisted channels.</p>';
    }

    const unresolvedItems = Array.isArray(result.unresolved) ? result.unresolved : [];
    if (unresolvedItems.length) {
      const listItems = unresolvedItems
        .map((item) => {
          const normalized = typeof item?.normalized === 'string' ? item.normalized : '';
          const original = typeof item?.input === 'string' ? item.input : '';
          const inputValueRaw = normalized || original;
          const inputValue = escapeHtml(inputValueRaw || '[empty]');
          const metaParts = [];
          if (typeof item?.row === 'number') {
            metaParts.push(`row ${item.row}`);
          }
          if (typeof item?.column === 'string' && item.column) {
            metaParts.push(item.column);
          }
          const metaLabel = metaParts.length
            ? `<span class="modal-summary__meta">${escapeHtml(metaParts.join(' · '))}</span>`
            : '';
          const messageText = typeof item?.message === 'string' && item.message
            ? escapeHtml(item.message)
            : 'Unable to resolve channel.';
          const reasonText =
            typeof item?.reason === 'string' && item.reason && item.reason !== item.message
              ? `<span class="modal-summary__reason">[${escapeHtml(item.reason)}]</span>`
              : '';
          return `
            <li class="modal-summary__item">
              <div class="modal-summary__item-header">⚠️ <code class="modal-summary__code">${inputValue}</code>${metaLabel}</div>
              <div class="modal-summary__message">${messageText}${reasonText}</div>
            </li>
          `;
        })
        .join('');
      const openAttr = unresolvedItems.length <= 3 ? ' open' : '';
      html += `
        <details class="modal-summary__details"${openAttr}>
          <summary>Review unresolved (${unresolvedItems.length})</summary>
          <ul class="modal-summary__list">${listItems}</ul>
        </details>
      `;
    }

    summaryEl.innerHTML = html;
    const classes = ['modal-summary'];
    if (counts.created > 0 || counts.updated > 0) {
      classes.push('modal-summary--success');
    } else if (counts.unresolved > 0) {
      classes.push('modal-summary--error');
    }
    summaryEl.className = classes.join(' ');
  }

  async handleBlacklistImport() {
    const file = this.el.modalFile.files?.[0];
    if (!file) {
      this.el.modalSummary.textContent = 'Please choose a CSV file to import.';
      this.el.modalSummary.className = 'modal-summary modal-summary--error';
      return;
    }
    this.el.modalSubmit.disabled = true;
    this.el.modalCancel.disabled = true;
    this.el.modalSummary.innerHTML = '<p class="modal-summary__headline">Importing…</p>';
    this.el.modalSummary.className = 'modal-summary';
    try {
      const result = await importBlacklist(file);
      this.renderBlacklistImportResult(result);
      await this.loadStats();
      await this.loadTable(Category.BLACKLISTED, (state) => ({ ...state, loading: true }));
      this.updateStatusBar('Blacklist import completed.', 'success');
    } catch (error) {
      console.error('Blacklist import failed', error);
      this.el.modalSummary.textContent = error instanceof Error ? error.message : 'Import failed.';
      this.el.modalSummary.className = 'modal-summary modal-summary--error';
    } finally {
      this.el.modalSubmit.disabled = false;
      this.el.modalCancel.disabled = false;
    }
  }
}

window.addEventListener('DOMContentLoaded', () => {
  const root = document.getElementById('app');
  if (!root) {
    throw new Error('App root not found');
  }
  const dashboard = new Dashboard(root);
  dashboard.init();
});

