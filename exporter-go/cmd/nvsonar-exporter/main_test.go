package main

import (
	"reflect"
	"testing"
)

func TestParseGPUs(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want []int
		err  bool
	}{
		{"", []int{0, 1, 2}, false},
		{"all", []int{0, 1, 2}, false},
		{"-1", []int{0, 1, 2}, false},
		{"1", []int{1}, false},
		{"0, 2", []int{0, 2}, false},
		{"3", nil, true},
		{"x", nil, true},
	} {
		got, err := parseGPUs(tc.in, 3)
		if (err != nil) != tc.err {
			t.Errorf("%q: err %v, want error %v", tc.in, err, tc.err)
			continue
		}
		if !tc.err && !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%q: got %v, want %v", tc.in, got, tc.want)
		}
	}
}
