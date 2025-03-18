-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY countingSim IS
    GENERIC (
        n           : INTEGER := 10;  -- Bit-widths of memory pointer, counter, etc.
        d           : INTEGER := 10;  -- Number of confComp module
        z           : INTEGER := 0;   -- Zero-padding to 2** for RSA
        classNumber : INTEGER := 10;  -- Class number for memory image
        logInNum    : INTEGER := 3    -- MuxCell, ceilingLOG2(#popCounters)
    );
    PORT (
        clk         : IN  STD_LOGIC;
        rst         : IN  STD_LOGIC;
        run         : IN  STD_LOGIC;   -- Run should be always '1' during calculation (ctrl)
        done        : IN  STD_LOGIC;
        reg1Update  : IN  STD_LOGIC;
        reg1rst     : IN  STD_LOGIC;
        reg2Update  : IN  STD_LOGIC;
        reg2rst     : IN  STD_LOGIC;
        muxSel      : IN  STD_LOGIC_VECTOR(logInNum DOWNTO 0);
        hv          : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
        CHV         : IN  STD_LOGIC_VECTOR(d-1 DOWNTO 0);
        pointer     : IN  STD_LOGIC_VECTOR(n-1 DOWNTO 0);
        dout        : OUT STD_LOGIC_VECTOR(n + logInNum - 1 DOWNTO 0)
    );
END ENTITY countingSim;

ARCHITECTURE behavioral OF countingSim IS

    -- Full Configuration Component Declaration
    COMPONENT fullconfComp IS
        GENERIC (
            n            : INTEGER := 10;  -- Bit-widths of memory pointer, counter, etc.
            classNumber  : INTEGER := 10;  -- Class number for memory image
            classPortion : INTEGER := 10   -- Portion of class memory for image
        );
        PORT (
            clk       : IN  STD_LOGIC;
            rst       : IN  STD_LOGIC;
            run       : IN  STD_LOGIC;
            done      : IN  STD_LOGIC;
            hv        : IN  STD_LOGIC;
            Chv_input : IN  STD_LOGIC;  
            pointer   : IN  STD_LOGIC_VECTOR(n-1 DOWNTO 0);
            sim       : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0)
        );
    END COMPONENT;

    -- RSA Component Declaration
    COMPONENT RSA IS
        GENERIC (
            inLen    : INTEGER := 8;   -- Bit width out popCounters
            d        : INTEGER := 8;   -- Number of popCounters
            z        : INTEGER := 0;   -- Zero-padding to 2**
            logInNum : INTEGER := 3    -- MuxCell, ceilingLOG2(#popCounters)
        );
        PORT (
            clk        : IN  STD_LOGIC;
            rst        : IN  STD_LOGIC;
            reg1Update : IN  STD_LOGIC;
            reg1rst    : IN  STD_LOGIC;
            reg2Update : IN  STD_LOGIC;
            reg2rst    : IN  STD_LOGIC;
            muxSel     : IN  STD_LOGIC_VECTOR(logInNum DOWNTO 0);
            A          : IN  STD_LOGIC_VECTOR((d * inLen) - 1 DOWNTO 0);
            B          : OUT STD_LOGIC_VECTOR(inLen + logInNum - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signal Declaration
    SIGNAL sim : STD_LOGIC_VECTOR((d * n) - 1 DOWNTO 0);

BEGIN

    -- Generate Array of fullconfComp Instances
    compArr: FOR I IN d DOWNTO 1 GENERATE
        comp : fullconfComp
            GENERIC MAP (
                n            => n,
                classNumber  => classNumber,
                classPortion => I - 1  -- Assign a unique portion value
            )
            PORT MAP (
                clk     => clk,
                rst     => rst,
                run     => run,
                done    => done,
                hv      => hv(I-1),
                Chv_input => CHV(I-1),
                pointer => pointer,
                sim     => sim((I * n) - 1 DOWNTO ((I - 1) * n))
            );
    END GENERATE compArr;

    -- RSA Instance with Explicit Generic Mapping
    seqAdd : RSA
        GENERIC MAP (
            inLen    => n,
            d        => d,
            z        => z,
            logInNum => logInNum
        )
        PORT MAP (
            clk        => clk,
            rst        => rst,
            reg1Update => reg1Update,
            reg1rst    => reg1rst,
            reg2Update => reg2Update,
            reg2rst    => reg2rst,
            muxSel     => muxSel,
            A          => sim,
            B          => dout
        );

END ARCHITECTURE behavioral;
