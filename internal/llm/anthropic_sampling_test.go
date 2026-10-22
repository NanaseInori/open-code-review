// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"testing/iotest"
	"time"

	"github.com/anthropics/anthropic-sdk-go/option"
)

func TestAnthropicClient_SamplingAPIVersions(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	zero, one := 0.0, 1.0
	tests := []struct {
		name        string
		version     string
		temperature *float64
		extraBody   map[string]any
		wantRemoved string
	}{
		{name: "default preserves sampling", temperature: &zero, extraBody: map[string]any{"top_p": 0.5, "top_k": 10}},
		{name: "legacy preserves sampling", version: "2023-06-01", temperature: &one, extraBody: map[string]any{"top_p": 0.5, "top_k": 10}},
		{name: "new without sampling", version: "2026-10-21", extraBody: map[string]any{"metadata": map[string]any{"user_id": "test-user"}, "temperature_override": 0.1}},
		{name: "typed zero", version: "2026-10-21", temperature: &zero, wantRemoved: "temperature"},
		{name: "typed default", version: "2026-10-21", temperature: &one, wantRemoved: "temperature"},
		{name: "version header whitespace", version: " 2026-10-21\t", temperature: &zero, wantRemoved: "temperature"},
		{name: "extra temperature", version: "2026-10-21", extraBody: map[string]any{"temperature": 0.5}, wantRemoved: "temperature"},
		{name: "extra zero temperature", version: "2026-10-21", extraBody: map[string]any{"temperature": 0}, wantRemoved: "temperature"},
		{name: "extra null temperature", version: "2026-10-21", extraBody: map[string]any{"temperature": nil}, wantRemoved: "temperature"},
		{name: "extra top_p", version: "2026-10-21", extraBody: map[string]any{"top_p": 0.5}, wantRemoved: "top_p"},
		{name: "extra zero top_p", version: "2026-10-21", extraBody: map[string]any{"top_p": 0}, wantRemoved: "top_p"},
		{name: "extra null top_p", version: "2026-10-21", extraBody: map[string]any{"top_p": nil}, wantRemoved: "top_p"},
		{name: "extra top_k", version: "2026-10-21", extraBody: map[string]any{"top_k": 10}, wantRemoved: "top_k"},
		{name: "extra zero top_k", version: "2026-10-21", extraBody: map[string]any{"top_k": 0}, wantRemoved: "top_k"},
		{name: "extra null top_k", version: "2026-10-21", extraBody: map[string]any{"top_k": nil}, wantRemoved: "top_k"},
		{name: "all removed", version: "2026-10-21", temperature: &zero, extraBody: map[string]any{"top_k": 10, "top_p": 0.5}, wantRemoved: "temperature, top_p, top_k"},
		{name: "JSON path creates removed field", version: "2026-10-21", extraBody: map[string]any{"temperature.value": 0}, wantRemoved: "temperature"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var calls atomic.Int32
			var gotVersion string
			var gotBody map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				gotVersion = r.Header.Get("anthropic-version")
				if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
					t.Errorf("decode request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, anthropicOKBody)
			}))
			defer server.Close()

			collector := NewRetryCollector()
			writer := &recordingRawWriter{}
			holder := &RawHolder{}
			holder.Set(writer)
			cfg := ClientConfig{
				URL: server.URL, APIKey: "test-key", Model: "claude-test",
				ExtraBody: tt.extraBody, RetryCodes: []int{400},
				retryCollector: collector, rawHolder: holder,
			}
			if tt.version != "" {
				cfg.ExtraHeaders = map[string]string{"aNtHrOpIc-VeRsIoN": tt.version}
			}
			ctx, cancel := context.WithTimeout(metaCtx(testMeta()), 2*time.Second)
			defer cancel()
			_, err := NewAnthropicClient(cfg).CompletionsWithCtx(ctx, ChatRequest{
				Messages: []Message{{Role: "user", Content: "ping"}}, Temperature: tt.temperature,
			})
			if tt.wantRemoved != "" {
				if err == nil {
					t.Fatal("expected local sampling validation error")
				}
				for _, want := range []string{"2026-10-21", tt.wantRemoved, "remove", "extra_body", "2023-06-01"} {
					if !strings.Contains(err.Error(), want) {
						t.Errorf("error %q does not contain %q", err, want)
					}
				}
				if calls.Load() != 0 {
					t.Errorf("server received %d requests, want none", calls.Load())
				}
				if attempts := attemptsFor(t, collector, testMeta()); len(attempts) != 0 {
					t.Errorf("local validation recorded %d HTTP attempts", len(attempts))
				}
				writer.mu.Lock()
				records := len(writer.records)
				writer.mu.Unlock()
				if records != 0 {
					t.Errorf("local validation recorded %d raw HTTP records", records)
				}
				return
			}
			if err != nil {
				t.Fatalf("CompletionsWithCtx: %v", err)
			}
			if calls.Load() != 1 {
				t.Fatalf("server received %d requests, want 1", calls.Load())
			}
			wantVersion := tt.version
			if wantVersion == "" {
				wantVersion = "2023-06-01"
			}
			if gotVersion != wantVersion {
				t.Errorf("API version = %q, want %q", gotVersion, wantVersion)
			}
			if tt.temperature != nil && gotBody["temperature"] != *tt.temperature {
				t.Errorf("temperature = %v, want %v", gotBody["temperature"], *tt.temperature)
			}
			for name, value := range tt.extraBody {
				encoded, _ := json.Marshal(value)
				var want any
				_ = json.Unmarshal(encoded, &want)
				if !reflect.DeepEqual(gotBody[name], want) {
					t.Errorf("%s = %v, want %v", name, gotBody[name], want)
				}
			}
		})
	}
}

func TestAnthropicClient_SamplingUsesEffectiveSDKVersion(t *testing.T) {
	for _, name := range []string{"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN_FILE", "ANTHROPIC_IDENTITY_TOKEN"} {
		t.Setenv(name, "")
	}
	t.Setenv("ANTHROPIC_CONFIG_DIR", t.TempDir())
	tests := []struct {
		name, sdkVersion, requestVersion string
		wantError                        bool
	}{
		{name: "SDK custom header", sdkVersion: "2026-10-21", wantError: true},
		{name: "request overrides new SDK header", sdkVersion: "2026-10-21", requestVersion: "2023-06-01"},
		{name: "request overrides legacy SDK header", sdkVersion: "2023-06-01", requestVersion: "2026-10-21", wantError: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("ANTHROPIC_CUSTOM_HEADERS", "Anthropic-Version: "+tt.sdkVersion)
			var calls atomic.Int32
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)
				if got := r.Header.Get("anthropic-version"); got != "2023-06-01" {
					t.Errorf("API version = %q, want legacy version", got)
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, anthropicOKBody)
			}))
			defer server.Close()
			cfg := ClientConfig{URL: server.URL, APIKey: "test-key", Model: "claude-test", ExtraBody: map[string]any{"top_p": 0.5}}
			if tt.requestVersion != "" {
				cfg.ExtraHeaders = map[string]string{"anthropic-version": tt.requestVersion}
			}
			_, err := ping(context.Background(), NewAnthropicClient(cfg))
			if tt.wantError {
				if err == nil || !strings.Contains(err.Error(), "top_p") || calls.Load() != 0 {
					t.Fatalf("error = %v, server calls = %d; want local top_p rejection", err, calls.Load())
				}
			} else if err != nil || calls.Load() != 1 {
				t.Fatalf("error = %v, server calls = %d; want legacy request", err, calls.Load())
			}
		})
	}
}

func TestAnthropicClient_SamplingValidationPreservesRetryBody(t *testing.T) {
	t.Setenv("ANTHROPIC_API_KEY", "test-key")
	var calls atomic.Int32
	bodies := make(chan []byte, 2)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request: %v", err)
		}
		bodies <- body
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			w.Header().Set("Retry-After-Ms", "1")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error","message":"retry"}}`)
			return
		}
		_, _ = io.WriteString(w, anthropicOKBody)
	}))
	defer server.Close()
	collector := NewRetryCollector()
	client := anthropicClient(server.URL, collector, map[string]string{"anthropic-version": "2026-10-21"})
	if _, err := ping(metaCtx(testMeta()), client); err != nil {
		t.Fatalf("CompletionsWithCtx: %v", err)
	}
	if calls.Load() != 2 {
		t.Fatalf("server received %d requests, want 2", calls.Load())
	}
	first, second := <-bodies, <-bodies
	if !json.Valid(first) || !bytes.Equal(first, second) {
		t.Errorf("retry changed request body: %s -> %s", first, second)
	}
	if attempts := freezeOne(t, collector).Attempts; len(attempts) != 2 {
		t.Errorf("recorded %d HTTP attempts, want 2", len(attempts))
	}
}

func TestAnthropicSamplingValidationLeavesOtherProtocolsUnchanged(t *testing.T) {
	for name, value := range map[string]string{"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret", "AWS_SESSION_TOKEN": "", "AWS_BEARER_TOKEN_BEDROCK": "", "AWS_EC2_METADATA_DISABLED": "true", "ANTHROPIC_API_KEY": "test-key"} {
		t.Setenv(name, value)
	}
	for _, protocol := range []string{"openai", "bedrock"} {
		t.Run(protocol, func(t *testing.T) {
			var gotBody map[string]any
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
					t.Errorf("decode request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				if protocol == "openai" {
					_, _ = io.WriteString(w, openAIOKBody)
				} else {
					_, _ = io.WriteString(w, anthropicOKBody)
				}
			}))
			defer server.Close()
			cfg := ClientConfig{
				URL: server.URL + "/v1", APIKey: "test-key", Model: "claude-test", AWSRegion: "us-east-1",
				ExtraHeaders: map[string]string{"anthropic-version": "2026-10-21"},
				ExtraBody:    map[string]any{"temperature": 0.5, "top_p": 0.9, "top_k": 10},
			}
			var client LLMClient
			if protocol == "bedrock" {
				bedrock := NewAnthropicBedrockClient(cfg)
				bedrock.sdk.Messages.Options = append(bedrock.sdk.Messages.Options, option.WithHTTPClient(&http.Client{
					Transport: resolverRoundTripFunc(func(r *http.Request) (*http.Response, error) {
						if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
							return nil, err
						}
						return &http.Response{
							StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"application/json"}},
							Body: io.NopCloser(strings.NewReader(anthropicOKBody)), Request: r,
						}, nil
					}),
				}))
				client = bedrock
			} else {
				client = NewOpenAIClient(cfg)
			}
			if _, err := ping(context.Background(), client); err != nil {
				t.Fatalf("CompletionsWithCtx: %v", err)
			}
			for name, want := range map[string]float64{"temperature": 0.5, "top_p": 0.9, "top_k": 10} {
				if gotBody[name] != want {
					t.Errorf("%s = %v, want %v", name, gotBody[name], want)
				}
			}
			if protocol == "bedrock" && gotBody["anthropic_version"] != "bedrock-2023-05-31" {
				t.Errorf("Bedrock version = %v, want unchanged legacy version", gotBody["anthropic_version"])
			}
		})
	}
}

func TestValidateAnthropicSamplingBodyErrorsAndNestedFields(t *testing.T) {
	tests := []struct {
		name, body, wantError string
	}{
		{name: "nested fields are preserved", body: `{"metadata":{"top_p":0},"temperature_override":0}`},
		{name: "invalid JSON", body: `{`, wantError: "decode Anthropic request body"},
		{name: "read failure", wantError: "read Anthropic request body"},
		{name: "no body"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
			req.Header.Set("anthropic-version", "2026-10-21")
			if tt.name == "read failure" {
				req.Body = io.NopCloser(iotest.ErrReader(errors.New("body unavailable")))
			} else if tt.body != "" {
				req.Body = io.NopCloser(strings.NewReader(tt.body))
			} else {
				req.Body = nil
			}
			called := false
			_, err := validateAnthropicSampling(req, func(r *http.Request) (*http.Response, error) {
				called = true
				if r.Body != nil {
					body, _ := io.ReadAll(r.Body)
					if string(body) != tt.body {
						t.Errorf("body changed: %q, want %q", body, tt.body)
					}
				}
				return &http.Response{StatusCode: http.StatusOK}, nil
			})
			if tt.wantError == "" {
				if err != nil || !called {
					t.Fatalf("error = %v, next called = %v; want pass-through", err, called)
				}
			} else if err == nil || !strings.Contains(err.Error(), tt.wantError) || called {
				t.Fatalf("error = %v, next called = %v; want local body error", err, called)
			}
		})
	}
}
