// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 alibaba/open-code-review Contributors

package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const anthropicSamplingRemovalVersion = "2026-10-21"

// Validate the serialized request so SDK defaults and extra_body JSON paths
// cannot bypass the check. Rejecting preserves explicit sampling choices.
func validateAnthropicSampling(req *http.Request, next func(*http.Request) (*http.Response, error)) (*http.Response, error) {
	if strings.TrimSpace(req.Header.Get("anthropic-version")) != anthropicSamplingRemovalVersion || req.Body == nil {
		return next(req)
	}

	body, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	req.Body = io.NopCloser(bytes.NewReader(body))
	if err != nil {
		return invalidAnthropicRequest(req, fmt.Errorf("read Anthropic request body: %w", err))
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return invalidAnthropicRequest(req, fmt.Errorf("decode Anthropic request body: %w", err))
	}
	var removed []string
	for _, name := range []string{"temperature", "top_p", "top_k"} {
		if _, present := fields[name]; present {
			removed = append(removed, name)
		}
	}
	if len(removed) > 0 {
		return invalidAnthropicRequest(req, fmt.Errorf("Anthropic API version %s does not support sampling parameters %s; remove them from the request and extra_body, or use anthropic-version: 2023-06-01", anthropicSamplingRemovalVersion, strings.Join(removed, ", ")))
	}
	return next(req)
}

func invalidAnthropicRequest(req *http.Request, err error) (*http.Response, error) {
	// The SDK retries nil responses even for local errors. Keep validation
	// outside HTTP-attempt observers and stop before any transport or backoff.
	return &http.Response{
		StatusCode: http.StatusBadRequest,
		Header:     http.Header{"X-Should-Retry": {"false"}},
		Body:       http.NoBody,
		Request:    req,
	}, err
}
