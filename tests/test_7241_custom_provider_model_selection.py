"""Regression coverage for #7241.

A model picked from a NON-ACTIVE named custom-provider group is exposed by
``/api/models`` with a provider-qualified routing id
(``@custom:omni:antigravity/gemini-3.7-flash-tiered``). The catalog population
path assigned that id to ``option.value`` without the bare-model metadata, so
``_modelStateForSelect()`` fell back to the routing id and a NEW session
persisted/sent it AS THE MODEL NAME — the backend then re-split it and answered
``404 No active credentials for provider: @custom:omni:antigravity``.

The dropdown must yield the bare model id plus a separately identified
``model_provider``, exactly like the option-synthesis path used by existing
sessions and CLI invocations.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS_PATH = REPO_ROOT / "static" / "ui.js"
UI_JS = UI_JS_PATH.read_text(encoding="utf-8")
NODE = shutil.which("node")

_DRIVER = r"""
const fs=require('fs');
const src=fs.readFileSync(process.argv[2],'utf8');
function extract(name){
  let start=src.indexOf('async function '+name+'(');
  if(start<0) start=src.indexOf('function '+name+'(');
  if(start<0) throw new Error('missing '+name);
  // Skip the parameter list first: default params (opts={}) contain braces.
  let i=src.indexOf('(',src.indexOf(name,start));
  let depth=0;
  for(;i<src.length;i++){
    if(src[i]==='(') depth++;
    else if(src[i]===')'&&--depth===0){ i++; break; }
  }
  i=src.indexOf('{',i);
  depth=0;
  for(;i<src.length;i++){
    if(src[i]==='{') depth++;
    else if(src[i]==='}'&&--depth===0) return src.slice(start,i+1);
  }
  throw new Error('unterminated '+name);
}

class Node {
  constructor(tag){
    this.tagName=String(tag).toUpperCase();
    this.children=[];
    this.dataset={};
    this.parentElement=null;
    this.textContent='';
    this.label='';
    this.id='';
    this._value='';
    this._selected=null;
    this.classList={contains:()=>false,add(){},remove(){},toggle(){}};
  }
  appendChild(child){child.parentElement=this;this.children.push(child);return child;}
  removeChild(child){this.children=this.children.filter(x=>x!==child);child.parentElement=null;}
  querySelectorAll(selector){
    if(selector==='optgroup') return this.children.filter(x=>x.tagName==='OPTGROUP');
    return [];
  }
  querySelector(selector){
    if(selector==='optgroup > option, option'){
      for(const child of this.children){
        if(child.tagName==='OPTGROUP'&&child.children.length) return child.children[0];
        if(child.tagName==='OPTION') return child;
      }
    }
    return null;
  }
  get options(){
    if(this.tagName!=='SELECT') return [];
    return this.children.flatMap(x=>x.tagName==='OPTGROUP'?x.children:[x]).filter(x=>x.tagName==='OPTION');
  }
  set innerHTML(value){ if(!value) this.children=[]; }
  get innerHTML(){ return ''; }
  _ownerSelect(){
    let node=this.parentElement;
    while(node&&node.tagName!=='SELECT') node=node.parentElement;
    return node;
  }
  get selected(){
    const sel=this._ownerSelect();
    return sel?sel._selected===this:!!this._selected;
  }
  set selected(flag){
    const sel=this._ownerSelect();
    if(!sel){ this._selected=!!flag; return; }
    if(flag){ sel._selected=this; sel._value=this.value; }
    else if(sel._selected===this){ sel._selected=null; }
  }
  get value(){ return this._value; }
  set value(next){
    this._value=String(next||'');
    if(this.tagName==='SELECT') this._selected=this.options.find(o=>o.value===this._value)||null;
  }
  get selectedOptions(){
    if(this.tagName!=='SELECT') return [];
    if(this._selected&&this._selected.value===this._value) return [this._selected];
    const hit=this.options.find(o=>o.value===this._value);
    return hit?[hit]:[];
  }
}

const MODELS_PAYLOAD={
  active_provider:'custom:main',
  default_model:'main-model',
  configured_model_badges:{},
  groups:[
    {provider:'main',provider_id:'custom:main',models:[{id:'main-model',label:'main-model'}]},
    {provider:'omni',provider_id:'custom:omni',models:[
      {id:'@custom:omni:antigravity/gemini-3.7-flash-tiered',label:'antigravity/gemini-3.7-flash-tiered'},
      {id:'@custom:omni:model-a:free',label:'model-a:free'},
    ]},
    {provider:'OpenRouter',provider_id:'openrouter',models:[
      {id:'@openrouter:vendor/some-model',label:'vendor/some-model'},
    ]},
  ],
};

const select=new Node('select');
select.id='modelSelect';
globalThis.window=globalThis;
globalThis.document={baseURI:'http://localhost/',createElement:tag=>new Node(tag)};
globalThis.location={href:'http://localhost/'};
globalThis.S={session:null};
globalThis.$=id=>(id==='modelSelect'?select:null);
globalThis._dynamicModelLabels={};
globalThis._modelDropdownRequestSeq=0;
globalThis._modelCatalogFallbackRetried=false;
globalThis._configuredModelBadges={};
globalThis._modelEndpointErrors={};
globalThis._activeProvider=null;
globalThis._defaultModel=null;
globalThis.getModelLabel=id=>id;
globalThis.syncModelChip=()=>{};
globalThis.renderModelDropdown=()=>{};
globalThis._positionModelDropdown=()=>{};
globalThis._fetchLiveModels=()=>{};
globalThis._redirectIfUnauth=()=>false;
globalThis.console={warn(){},debug(){},log(){}};
globalThis.fetch=async()=>({status:200,ok:true,json:async()=>JSON.parse(JSON.stringify(MODELS_PAYLOAD))});

for(const name of [
  '_getOptionProviderId','_providerFromModelValue','_bareModelForProviderValue',
  '_modelPickerOptionIdentity','_deduplicateModelPickerOptions','_modelStateForSelect',
  '_captureModelDropdownSelection','_findModelInDropdown','_refreshOpenModelDropdown',
  '_applyModelToDropdown','_ensureModelOptionInDropdown','_reconcileModelDropdownSelection',
  'populateModelDropdown',
]) eval(extract(name));

(async()=>{
  await populateModelDropdown({});
  const ROUTED='@custom:omni:antigravity/gemini-3.7-flash-tiered';
  const catalogOption=select.options.find(o=>o.value===ROUTED);
  const optionCountAfterPopulate=select.options.length;

  // The composer picker hands the row's VALUE + the group provider id to
  // selectModelFromDropdown(), which delegates to _ensureModelOptionInDropdown.
  _ensureModelOptionInDropdown(ROUTED,select,'custom:omni');
  const newSessionState=_modelStateForSelect(select,select.value);
  const capturedState=_captureModelDropdownSelection(select);
  const optionCountAfterSelect=select.options.length;

  // Rehydration round-trip: the bare model + provider must resolve back to the
  // same routed option (persisted state / session reload / catalog rebuild).
  const reapplied=_applyModelToDropdown(newSessionState.model,select,newSessionState.model_provider);
  const reappliedState=_modelStateForSelect(select,select.value);

  _ensureModelOptionInDropdown('@custom:omni:model-a:free',select,'custom:omni');
  const colonModelState=_modelStateForSelect(select,select.value);

  _ensureModelOptionInDropdown('@openrouter:vendor/some-model',select,'openrouter');
  const openRouterState=_modelStateForSelect(select,select.value);

  // Values with no matching option keep the pre-existing fallback (no blind
  // re-parse of an unknown provider hint).
  const unknownState=_modelStateForSelect(select,'@custom:ghost:model-x:tier');

  // Settings / cron / profile pickers (panels.js) build their options straight
  // from /api/models WITHOUT data-model, so extraction alone has to recover the
  // bare id there.
  const legacySelect=new Node('select');
  legacySelect.id='settingsModel';
  const legacyGroup=new Node('optgroup');
  legacyGroup.dataset.provider='custom:omni';
  legacySelect.appendChild(legacyGroup);
  const legacyOption=new Node('option');
  legacyOption.value=ROUTED;
  legacyGroup.appendChild(legacyOption);
  legacySelect.value=ROUTED;
  const legacyState=_modelStateForSelect(legacySelect,legacySelect.value);

  // Dropdown UI rendering: entry.id must be the clean bare model ID, not the routing value.
  const omniEntryDisplayId=(catalogOption.dataset&&catalogOption.dataset.model)
    || ((typeof _bareModelForProviderValue==='function')?_bareModelForProviderValue(catalogOption.value,'custom:omni'):'')
    || catalogOption.value;

  process.stdout.write(JSON.stringify({
    catalogOptionDatasetModel:catalogOption?(catalogOption.dataset.model||null):null,
    catalogOptionProvider:catalogOption?_getOptionProviderId(catalogOption):null,
    omniEntryDisplayId,
    newSessionState,capturedState,reapplied,reappliedState,
    colonModelState,openRouterState,unknownState,legacyState,
    optionCountAfterPopulate,optionCountAfterSelect,
    optionValues:select.options.map(o=>o.value),
  }));
})().catch(err=>{
  process.stderr.write(String(err&&err.stack||err));
  process.exit(1);
});
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    if NODE is None:
        pytest.skip("node not in PATH")
    driver = tmp_path_factory.mktemp("issue7241") / "driver.js"
    driver.write_text(_DRIVER, encoding="utf-8")
    proc = subprocess.run(
        [NODE, str(driver), str(UI_JS_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_catalog_option_keeps_bare_model_metadata(result):
    """The population path must tag routed options with their bare model id."""
    assert result["catalogOptionDatasetModel"] == "antigravity/gemini-3.7-flash-tiered"
    assert result["catalogOptionProvider"] == "custom:omni"


def test_dropdown_row_displays_bare_model_id(result):
    """The dropdown row ID subtext must show the clean bare model ID."""
    assert result["omniEntryDisplayId"] == "antigravity/gemini-3.7-flash-tiered"


def test_new_session_selection_sends_bare_model_and_provider(result):
    """The whole point of #7241: no @provider:model string in the model field."""
    assert result["newSessionState"] == {
        "model": "antigravity/gemini-3.7-flash-tiered",
        "model_provider": "custom:omni",
    }
    assert not result["newSessionState"]["model"].startswith("@")


def test_persisted_dropdown_capture_matches_send_shape(result):
    """localStorage/new-session capture goes through the same extraction."""
    assert result["capturedState"] == {
        "model": "antigravity/gemini-3.7-flash-tiered",
        "model_provider": "custom:omni",
    }


def test_selection_reuses_the_catalog_option_instead_of_appending_a_twin(result):
    assert result["optionCountAfterSelect"] == result["optionCountAfterPopulate"]
    assert result["optionValues"].count("@custom:omni:antigravity/gemini-3.7-flash-tiered") == 1


def test_bare_model_round_trips_back_to_the_routed_option(result):
    """Rehydration must still resolve the routed option, not drop the provider."""
    assert result["reapplied"] == "@custom:omni:antigravity/gemini-3.7-flash-tiered"
    assert result["reappliedState"] == {
        "model": "antigravity/gemini-3.7-flash-tiered",
        "model_provider": "custom:omni",
    }


def test_colon_bearing_model_id_is_not_missplit(result):
    """#6221 guard: only the exact @provider: wrapper is stripped."""
    assert result["colonModelState"] == {
        "model": "model-a:free",
        "model_provider": "custom:omni",
    }


def test_plain_provider_prefix_also_yields_bare_model(result):
    assert result["openRouterState"] == {
        "model": "vendor/some-model",
        "model_provider": "openrouter",
    }


def test_unmatched_value_keeps_legacy_fallback(result):
    """No option to trust ⇒ no speculative strip of an unverified prefix."""
    assert result["unknownState"]["model"] == "@custom:ghost:model-x:tier"


def test_option_without_data_model_still_yields_bare_model(result):
    """panels.js pickers omit data-model; extraction must still strip the wrapper."""
    assert result["legacyState"] == {
        "model": "antigravity/gemini-3.7-flash-tiered",
        "model_provider": "custom:omni",
    }


def test_bare_model_helper_strips_only_the_exact_provider_wrapper():
    assert "function _bareModelForProviderValue(modelId, providerId){" in UI_JS
    assert "return value.toLowerCase().startsWith(prefix.toLowerCase())?value.slice(prefix.length):value;" in UI_JS


def test_selected_row_still_matches_when_state_model_is_bare():
    """The picker row value stays routed, so the highlight needs the sel.value arm."""
    assert (
        "||String((m&&m.value)||'')===String((sel&&sel.value)||''))"
        "&&String(_modelProviderForSelectedBadge(m)||'')" in UI_JS
    )
