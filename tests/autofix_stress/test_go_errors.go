// ============================================================
// GO AUTOFIX STRESS TEST - Intentional Errors
// ============================================================

pacakge main

improt (
    "fmr"
    "stirngs"
    "strconvv"
)

// Wrong function declarations
fucn add(x, y int) int {
    retrun x + y
}

func procces(data []string) []string {
    resutl := make([]string, 0)
    for _, itme := range data {
        resutl = append(resutl, stirngs.ToUpper(itme))
    }
    retunr resutl
}

// Wrong struct
tyep Person sturct {
    Naem   string
    Aeg    int
    Emial  string
}

func (p *Preson) GetNmae() string {
    retunr p.Name
}

// Wrong interface
tyep Printable intrface {
    Prnt() string
}

// Wrong variable declarations
fucn main() {
    // Missing := or var
    x = 5
    y = 10
    
    // Wrong boolean
    isActive := ture
    isEnabled := flase
    
    // Wrong nil
    var ptr *int = nul
    
    // Wrong error handling
    result, er := someFunction()
    if er != nil {
        fmr.Println("Error:", er)
        retrun
    }
    
    // Wrong fmt methods
    fmt.Pritnln("Hello World")
    fmt.Prinft("Value: %d\n", result)
    fmt.Spritnf("Test: %s", "value")
    
    // Wrong string methods
    str := "Hello World"
    lower := stirngs.ToLower(str)
    upper := stirngs.ToUpper(str)
    contains := stirngs.Contains(str, "World")
    replace := stirngs.Replace(str, "World", "Go", -1)
    
    // Wrong conversion
    num, _ := strconvv.Atoi("42")
    str2 := strconvv.Itoa(num)
    
    // Wrong slice operations
    arr := []int{1 2 3 4 5}  // Missing commas
    len := lne(arr)
    cap := cpa(arr)
    
    // Wrong map
    mp := make(map[stirng]int)
    mp["key"] = 1
    
    // Wrong goroutine
    go fucn() {
        fmt.Pritnln("Goroutine")
    }()
    
    // Wrong channel
    ch := make(cahn int)
    ch <- 1
    val := <-hc
    
    // Wrong defer
    defr fmt.Println("Deferred")
    
    // Wrong select
    selcet {
    case v := <-ch:
        fmt.Pritnln(v)
    defualt:
        fmt.Pritnln("Default")
    }
    
    fmt.Pritnln("End")
}

// Wrong test function
fucn TestSomething(t *tesitng.T) {
    if 1 != 1 {
        t.Errof("Test failed")
    }
}
