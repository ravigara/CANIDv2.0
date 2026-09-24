const state = {
  mode: "identify", identifyFile: null, registerFiles: [],
  registryDogs: [], selectedDogIds: new Set(),
};

const $ = (selector) => document.querySelector(selector);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
}[char]));
const pct = (value, formatted) => formatted || `${(Number(value || 0) * 100).toFixed(1)}%`;

function initials(name) {
  return String(name || "?").trim().split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "?";
}

function photoMarkup(url, name, className) {
  const fallback = `<span class="photo-placeholder">${esc(initials(name))}</span>`;
  if (!url) return `<div class="${className}">${fallback}</div>`;
  return `<div class="${className}"><img src="${esc(url)}" alt="${esc(name || "Dog")}" onerror="this.parentElement.innerHTML='${fallback.replace(/'/g, "\\'")}'"></div>`;
}

function setBusy(active, text) {
  $("#busy").classList.toggle("hidden", !active);
  if (text) $("#busy-text").textContent = text;
  document.querySelectorAll("button").forEach((button) => { button.disabled = active; });
}

function showResult(title, html) {
  $("#result-title").textContent = title;
  $("#result-content").innerHTML = html;
  $("#result-section").classList.remove("hidden");
  $("#result-section").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderError(message) {
  showResult("Something needs attention", `<div class="error-result">${esc(message)}</div>`);
}

function renderRegistration(data) {
  const dog = data.dog || {};
  showResult("Dog registered", `
    <div class="result-card recognized">
      ${photoMarkup(dog.photo_url, dog.name, "result-photo")}
      <div class="result-body">
        <div class="result-label">● Registered successfully</div>
        <h3>${esc(dog.name || dog.dog_id)}</h3>
        <p>${esc(dog.breed || "Profile created")} · ID ${esc(dog.dog_id)}</p>
        <div class="confidence"><strong>${esc(data.photos_processed)}</strong> usable photos <span>·</span> <strong>${esc(data.gallery_size)}</strong> dogs in gallery</div>
        <div class="result-meta">Per-feature templates saved for the rhinarium, nares and philtrum where detected.</div>
      </div>
    </div>`);
}

function renderCandidates(candidates) {
  if (!candidates.length) return `<div class="empty-result">No registered dog was close enough to display as a possible match.</div>`;
  return `<div class="candidate-list">${candidates.map((candidate, index) => {
    const dog = candidate.dog || {};
    const name = dog.name || candidate.dog_id || "Unknown dog";
    return `<div class="candidate">
      <span class="candidate-rank">${index + 1}</span>
      ${photoMarkup(dog.photo_url, name, "candidate-photo")}
      <div class="candidate-info"><strong>${esc(name)}</strong><small>${esc(dog.breed || dog.dog_id || "Registered profile")}</small></div>
      <span class="candidate-score">${esc(pct(candidate.confidence, candidate.confidence_pct))}</span>
    </div>`;
  }).join("")}</div>`;
}

function renderIdentification(data) {
  if (data.status === "no_nose") {
    showResult("Nose not detected", `<div class="empty-result">${esc(data.message || "Use a closer, clearer nose photo and try again.")}</div>`);
    return;
  }
  if (data.match && data.dog) {
    const dog = data.dog;
    showResult("Dog recognized", `
      <div class="result-card recognized">
        ${photoMarkup(dog.photo_url, dog.name, "result-photo")}
        <div class="result-body">
          <div class="result-label">● Registered dog found</div>
          <h3>${esc(dog.name || dog.dog_id)}</h3>
          <p>${esc(dog.breed || "Registered profile")} · ID ${esc(dog.dog_id)}</p>
          <div class="confidence"><strong>${esc(pct(data.confidence, data.confidence_pct))}</strong> match confidence</div>
          <div class="result-meta">${esc((data.features_used || []).length)} anatomical features used · margin ${esc(pct(data.margin))}</div>
        </div>
      </div>`);
    return;
  }
  const candidates = data.possible_matches || data.candidates || [];
  showResult("Not recognized", `
    <div class="result-card not-recognized">
      <div class="result-body"><div class="result-label warn">● No confident match</div><h3>Dog not found</h3><p>${esc(data.message || "This dog is not registered yet, or the evidence is not confident enough.")}</p></div>
      <div><p class="eyebrow">POSSIBLE MATCHES</p>${renderCandidates(candidates)}</div>
  </div>`);
}

function renderDogRegistryLegacy(dogs) {
  const grid = $("#dog-grid");
  if (!dogs.length) {
    grid.innerHTML = `<div class="registry-empty">No registered dogs yet. Register a dog above to create the first profile.</div>`;
    return;
  }
  grid.innerHTML = dogs.map((dog) => `
    <button class="dog-card" type="button" data-dog-id="${esc(dog.dog_id)}">
      ${photoMarkup(dog.photo_url, dog.name, "dog-card-photo")}
      <span class="dog-card-body"><strong>${esc(dog.name || dog.dog_id)}</strong><small>${esc(dog.breed || dog.dog_id)} · ${esc(dog.image_count || 0)} photos</small></span>
    </button>`).join("");
  grid.querySelectorAll(".dog-card").forEach((card) => {
    card.addEventListener("click", () => {
      const dog = dogs.find((item) => item.dog_id === card.dataset.dogId);
      if (dog) showDogDetail(dog);
    });
  });
}

function renderDogRegistry(dogs) {
  state.registryDogs = dogs;
  const knownIds = new Set(dogs.map((dog) => dog.dog_id));
  state.selectedDogIds = new Set([...state.selectedDogIds].filter((id) => knownIds.has(id)));
  const grid = $("#dog-grid");
  const tools = $("#registry-tools");
  if (!dogs.length) {
    tools.classList.add("hidden");
    grid.innerHTML = `<div class="registry-empty">No registered dogs yet. Register a dog above to create the first profile.</div>`;
    updateBulkControls();
    return;
  }
  tools.classList.remove("hidden");
  grid.innerHTML = dogs.map((dog) => `
    <article class="dog-card" data-dog-id="${esc(dog.dog_id)}">
      <label class="dog-select" title="Select ${esc(dog.name || dog.dog_id)}">
        <input class="dog-select-checkbox" type="checkbox" data-dog-id="${esc(dog.dog_id)}"${state.selectedDogIds.has(dog.dog_id) ? " checked" : ""}>
        <span>Select</span>
      </label>
      <button class="dog-card-open" type="button">
        ${photoMarkup(dog.photo_url, dog.name, "dog-card-photo")}
        <span class="dog-card-body"><strong>${esc(dog.name || dog.dog_id)}</strong><small>${esc(dog.breed || dog.dog_id)} · ${esc(dog.image_count || 0)} photos</small></span>
      </button>
    </article>`).join("");
  grid.querySelectorAll(".dog-select-checkbox").forEach((checkbox) => {
    checkbox.addEventListener("change", (event) => {
      const id = event.currentTarget.dataset.dogId;
      if (event.currentTarget.checked) state.selectedDogIds.add(id);
      else state.selectedDogIds.delete(id);
      updateBulkControls();
    });
  });
  grid.querySelectorAll(".dog-card-open").forEach((button) => {
    button.addEventListener("click", () => {
      const card = button.closest(".dog-card");
      const dog = dogs.find((item) => item.dog_id === card.dataset.dogId);
      if (dog) showDogDetail(dog);
    });
  });
  updateBulkControls();
}

function updateBulkControls() {
  const dogs = state.registryDogs;
  const selected = state.selectedDogIds.size;
  const selectAll = $("#select-all-dogs");
  const deleteButton = $("#delete-selected-dogs");
  if (!selectAll || !deleteButton) return;
  selectAll.checked = dogs.length > 0 && selected === dogs.length;
  selectAll.indeterminate = selected > 0 && selected < dogs.length;
  selectAll.disabled = !dogs.length;
  deleteButton.disabled = selected === 0;
  $("#selection-count").textContent = `${selected} selected`;
}

function showDogDetail(dog) {
  const images = dog.photo_urls || (dog.photo_url ? [dog.photo_url] : []);
  const metadata = [
    ["Chip ID", dog.identification_chip_id],
    ["Colour", dog.colour],
    ["Breed", dog.breed],
    ["Age", dog.age],
    ["Blood type", dog.blood_type],
  ].filter((item) => item[1]);
  const owner = dog.owner || {};
  const ownerDetails = [owner.name, owner.phone, owner.email, owner.address].filter(Boolean);
  const metadataMarkup = metadata.length
    ? `<div class="dog-detail-meta">${metadata.map(([label, value]) => `<span><small>${esc(label)}</small><strong>${esc(value)}</strong></span>`).join("")}</div>`
    : "";
  const ownerMarkup = ownerDetails.length
    ? `<div class="dog-owner"><p class="eyebrow">OWNER DETAILS</p><p>${ownerDetails.map((value) => esc(value)).join(" · ")}</p></div>`
    : "";
  const imageMarkup = images.length
    ? `<div class="dog-image-grid">${images.map((url, index) => `<figure><img src="${esc(url)}" alt="${esc(dog.name || dog.dog_id)} photo ${index + 1}"></figure>`).join("")}</div>`
    : `<div class="dog-detail-empty">No registration photos are available for this profile.</div>`;
  const detail = $("#dog-detail");
  detail.innerHTML = `
    <div class="dog-detail-header">
      <div><h3>${esc(dog.name || dog.dog_id)}</h3><p>${esc(dog.breed || "Registered profile")} · ID ${esc(dog.dog_id)} · ${esc(images.length)} photos</p></div>
      <div class="dog-detail-actions"><button class="dog-detail-update" type="button">Update images</button><button class="dog-detail-delete" type="button">Delete dog</button><button class="dog-detail-close" type="button">Close</button></div>
    </div>${metadataMarkup}${ownerMarkup}
    <form class="dog-update-form">
      <p class="eyebrow">UPDATE IDENTITY IMAGES</p>
      <p class="dog-update-copy">Add one or more clear photos of this dog’s current nose. Existing registration photos stay in the profile while the identity template is rebuilt from the full set.</p>
      <label class="dog-update-dropzone" for="dog-update-files">
        <input id="dog-update-files" name="files" type="file" accept="image/*" multiple capture="environment" required>
        <span class="upload-glyph">＋</span>
        <strong>Add current nose photos</strong>
        <span>Use a few angles and lighting conditions when possible</span>
        <span class="file-name">No new photos selected</span>
      </label>
      <button class="secondary-button" type="submit">Update identity images <span>→</span></button>
    </form>
    ${imageMarkup}`;
  detail.classList.remove("hidden");
  detail.querySelector(".dog-detail-close").addEventListener("click", () => detail.classList.add("hidden"));
  detail.querySelector(".dog-detail-delete").addEventListener("click", () => deleteDog(dog));
  detail.querySelector(".dog-detail-update").addEventListener("click", () => {
    detail.querySelector(".dog-update-form").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });
  detail.querySelector(".dog-update-form").addEventListener("submit", (event) => updateDogImages(event, dog));
  detail.querySelector("#dog-update-files").addEventListener("change", (event) => {
    const files = [...event.currentTarget.files];
    event.currentTarget.closest("label").querySelector(".file-name").textContent = files.length
      ? `${files.length} new photo${files.length === 1 ? "" : "s"} selected`
      : "No new photos selected";
  });
  detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function updateDogImages(event, dog) {
  event.preventDefault();
  const files = [...event.currentTarget.querySelector("input[type=file]").files];
  if (!files.length) return renderError("Choose at least one current photo first.");
  setBusy(true, "Rebuilding this dog’s identity template from the updated photos…");
  try {
    const form = new FormData(event.currentTarget);
    const response = await parseResponse(await fetch(`/api/dogs/${encodeURIComponent(dog.dog_id)}/images`, {
      method: "POST", body: form,
    }));
    showResult("Dog images updated", `<div class="empty-result success-result">${esc(response.message || "The dog’s identity images were updated.")} The next identification will use the updated template.</div>`);
    await refreshStatus();
    await refreshDogs();
    const updatedDog = state.registryDogs.find((item) => item.dog_id === dog.dog_id);
    if (updatedDog) showDogDetail(updatedDog);
  } catch (error) { renderError(error.message); }
  finally { setBusy(false); }
}

async function deleteDog(dog) {
  const name = dog.name || dog.dog_id;
  if (!window.confirm(`Delete ${name} from the active registry? Its stored registry photos will also be deleted.`)) return;
  setBusy(true, "Deleting the registered profile…");
  try {
    await parseResponse(await fetch(`/api/dogs/${encodeURIComponent(dog.dog_id)}`, { method: "DELETE" }));
    state.selectedDogIds.delete(dog.dog_id);
    $("#dog-detail").classList.add("hidden");
    showResult("Dog deleted", `<div class="empty-result">${esc(name)} and its active registry photos were deleted. Original source folders were not changed.</div>`);
    await refreshStatus();
    await refreshDogs();
  } catch (error) { renderError(error.message); }
  finally { setBusy(false); }
}

async function deleteSelectedDogs() {
  const selected = state.registryDogs.filter((dog) => state.selectedDogIds.has(dog.dog_id));
  if (!selected.length) return;
  const names = selected.map((dog) => dog.name || dog.dog_id).join(", ");
  if (!window.confirm(`Delete ${selected.length} selected dog${selected.length === 1 ? "" : "s"} (${names}) from the active registry? Their stored registry photos will also be deleted.`)) return;
  setBusy(true, `Deleting ${selected.length} registered profiles…`);
  try {
    const response = await parseResponse(await fetch("/api/dogs/bulk-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dog_ids: selected.map((dog) => dog.dog_id) }),
    }));
    state.selectedDogIds.clear();
    $("#dog-detail").classList.add("hidden");
    showResult("Dogs deleted", `<div class="empty-result">${esc(response.deleted_count)} selected profiles and their active registry photos were deleted. Original source folders were not changed.</div>`);
    await refreshStatus();
    await refreshDogs();
  } catch (error) { renderError(error.message); }
  finally { setBusy(false); updateBulkControls(); }
}

async function refreshDogs() {
  try {
    const data = await parseResponse(await fetch("/api/dogs"));
    renderDogRegistry(data.dogs || []);
  } catch (error) {
    state.registryDogs = [];
    state.selectedDogIds.clear();
    $("#dog-grid").innerHTML = `<div class="registry-empty">${esc(error.message)}</div>`;
    updateBulkControls();
  }
}

async function parseResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "The server could not complete that request.");
  return data;
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode-tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  $("#identify-form").classList.toggle("hidden", mode !== "identify");
  $("#register-form").classList.toggle("hidden", mode !== "register");
  $("#mode-kicker").textContent = mode === "identify" ? "IDENTIFICATION" : "REGISTRATION";
  $("#mode-title").textContent = mode === "identify" ? "Who is this dog?" : "Create a dog profile";
  $("#mode-description").textContent = mode === "identify"
    ? "Upload one sharp, front-facing nose photo. The system checks the detected anatomy against the registered feature gallery."
    : "Add a name and several clear nose photos. The system stores independent templates for each detected nose feature.";
}

function previewIdentify(file) {
  if (!file) return;
  state.identifyFile = file;
  $("#identify-file-name").textContent = file.name;
  $("#identify-preview").src = URL.createObjectURL(file);
  $("#identify-preview-wrap").classList.remove("hidden");
}

function previewRegister(files) {
  state.registerFiles = [...files];
  $("#register-file-name").textContent = files.length ? `${files.length} photo${files.length === 1 ? "" : "s"} selected` : "No photos selected";
  $("#register-previews").innerHTML = state.registerFiles.map((file) => `<img src="${URL.createObjectURL(file)}" alt="Selected registration photo">`).join("");
}

async function submitIdentify(event) {
  event.preventDefault();
  if (!state.identifyFile) return renderError("Choose a nose photo first.");
  setBusy(true, "Detecting anatomy and comparing feature embeddings…");
  try {
    const form = new FormData();
    form.append("file", state.identifyFile);
    renderIdentification(await parseResponse(await fetch("/api/identify", { method: "POST", body: form })));
  } catch (error) { renderError(error.message); }
  finally { setBusy(false); }
}

async function submitRegister(event) {
  event.preventDefault();
  if (state.registerFiles.length < 3) return renderError("Please select at least three nose photos for registration.");
  setBusy(true, "Building per-feature identity templates…");
  try {
    const form = new FormData(event.currentTarget);
    renderRegistration(await parseResponse(await fetch("/api/register", { method: "POST", body: form })));
    await refreshStatus();
    await refreshDogs();
  } catch (error) { renderError(error.message); }
  finally { setBusy(false); }
}

async function refreshStatus() {
  try {
    const data = await parseResponse(await fetch("/api/health"));
    $("#system-status").innerHTML = `<span class="status-dot ready"></span>${esc(data.gallery_size)} registered dogs · models ready`;
    $("#gallery-count").textContent = `${data.gallery_size} registered dogs · ${data.embedding_version}`;
  } catch (error) {
    $("#system-status").innerHTML = `<span class="status-dot error"></span>Gallery unavailable`;
    $("#gallery-count").textContent = error.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".mode-tab").forEach((tab) => tab.addEventListener("click", () => setMode(tab.dataset.mode)));
  $("#identify-file").addEventListener("change", (event) => previewIdentify(event.target.files[0]));
  $("#register-files").addEventListener("change", (event) => previewRegister(event.target.files));
  $("#identify-form").addEventListener("submit", submitIdentify);
  $("#register-form").addEventListener("submit", submitRegister);
  $("#clear-result").addEventListener("click", () => $("#result-section").classList.add("hidden"));
  $("#refresh-dogs").addEventListener("click", refreshDogs);
  $("#select-all-dogs").addEventListener("change", (event) => {
    state.selectedDogIds = event.currentTarget.checked
      ? new Set(state.registryDogs.map((dog) => dog.dog_id))
      : new Set();
    renderDogRegistry(state.registryDogs);
  });
  $("#delete-selected-dogs").addEventListener("click", deleteSelectedDogs);
  refreshStatus();
  refreshDogs();
});
